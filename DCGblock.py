import torch
import torch.nn as nn
import numpy as np  
import torch.distributed as dist
from .math_utils import symmetric_binary_kl, binary_js_divergence, js_divergence


# Compute discrepancy map
def compute_diff_map(teacher1_logits, teacher2_logits, use_sigmoid):
    if use_sigmoid:
        # Region-based / Sigmoid mode using Binary JS Divergence.
        # JS divergence is bounded [0, ln2] and more stable.
        t1_probs = torch.sigmoid(teacher1_logits)
        t2_probs = torch.sigmoid(teacher2_logits)
        diff_map_raw = binary_js_divergence(t1_probs, t2_probs)
        
        if diff_map_raw.shape[1] > 1:
            # Average over channels to get single-channel uncertainty map
            diff_map = diff_map_raw.mean(dim=1, keepdim=True)
        else:
            diff_map = diff_map_raw
    else:
        # Multi-class / Softmax mode using JS Divergence
        t1_probs = torch.softmax(teacher1_logits, dim=1)
        t2_probs = torch.softmax(teacher2_logits, dim=1)
        diff_map = js_divergence(t1_probs, t2_probs)
    return diff_map


##################################### Part 1: Fixed Threshold #####################################
# Retains pixels with discrepancy less than a predefined threshold.

class ConsensusMaskGenerator_Threshold(nn.Module):
    def __init__(self, threshold=0.1, use_sigmoid=True):
        super().__init__()
        self.threshold = threshold
        self.use_sigmoid = use_sigmoid
        # Logging variables
        self.current_ratio = 0.0 
        self.current_kappa = threshold 

    def update_kappa(self, epoch: int):
        # Threshold mode does not change with epochs
        pass

    def forward(self, teacher1_logits, teacher2_logits) -> torch.Tensor:
        diff_map = compute_diff_map(teacher1_logits, teacher2_logits, self.use_sigmoid)

        # Core logic: absolute threshold masking
        consensus_mask = (diff_map <= self.threshold).float()
        
        # Update logging variables
        with torch.no_grad():
            self.current_ratio = consensus_mask.mean().item()
            
        return consensus_mask


##################################### Part 2: TopK Ratio #####################################
# Retains a top proportion (Ratio) of the most consistent pixels per epoch.

class ConsensusMaskGenerator_TopK(nn.Module):
    def __init__(self, kappa_start=0.5, kappa_end=0.98, rampup_epochs=50, total_epochs=200, use_sigmoid=True):
        """
        :param rampup_epochs: Epochs to complete kappa ramp-up.
        """
        super().__init__()
        self.kappa_start = kappa_start
        self.kappa_end = kappa_end
        self.rampup_epochs = rampup_epochs
        # Backward compatibility for total_epochs
        self.total_epochs = total_epochs 
        
        self.current_ratio = kappa_start 
        self.current_kappa = kappa_start
        self.use_sigmoid = use_sigmoid

    def update_kappa(self, epoch: int):
        # Use rampup_epochs for fast warmup
        if self.rampup_epochs > 0:
            progress = min(1.0, epoch / self.rampup_epochs)
        else:
            progress = 1.0
            
        self.current_ratio = self.kappa_start + (self.kappa_end - self.kappa_start) * progress
        self.current_kappa = self.current_ratio

    def forward(self, teacher1_logits, teacher2_logits) -> torch.Tensor:
        diff_map = compute_diff_map(teacher1_logits, teacher2_logits, self.use_sigmoid)

        B, C, D, H, W = diff_map.shape
        diff_flat = diff_map.view(B, -1)

        # Core logic: TopK ranking masking
        k = int(diff_flat.shape[1] * self.current_ratio)
        k = max(1, min(k, diff_flat.shape[1]))
        
        # Find the k-th smallest diff value as the temporary threshold
        vals, _ = torch.topk(diff_flat, k, dim=1, largest=False, sorted=False)
        threshold = vals.max(dim=1, keepdim=True)[0]

        consensus_mask = (diff_map <= threshold.view(B, 1, 1, 1, 1)).float()
        
        return consensus_mask


##################################### Part 3: Dynamic Threshold #####################################
# Automatically computes threshold based on batch Mean and Std of discrepancies.
# Threshold = Mean - sigma * Std

class ConsensusMaskGenerator_Dynamic(nn.Module):
    def __init__(self, sigma=0.5, use_sigmoid=True):
        """
        :param sigma: Margin coefficient. Threshold = Mean - sigma * Std.
        """
        super().__init__()
        self.sigma = sigma
        self.use_sigmoid = use_sigmoid
        self.current_ratio = 0.0
        self.current_kappa = 0.0 # Refers to the dynamically calculated threshold

    def update_kappa(self, epoch: int):
        # Fully adaptive, no update needed
        pass

    def forward(self, teacher1_logits, teacher2_logits) -> torch.Tensor:
        diff_map = compute_diff_map(teacher1_logits, teacher2_logits, self.use_sigmoid)
        
        # Flatten for statistics
        diff_flat = diff_map.view(-1)
        
        # Core logic: Dynamic thresholding based on distribution statistics
        mean_val = diff_flat.mean()
        std_val = diff_flat.std()
        
        # Compute dynamic threshold: keep pixels with discrepancy significantly lower than average.
        dynamic_threshold = mean_val - self.sigma * std_val
        
        # Add a small epsilon to prevent completely empty masks
        dynamic_threshold = torch.max(dynamic_threshold, torch.tensor(1e-6, device=diff_map.device))

        consensus_mask = (diff_map <= dynamic_threshold).float()
        
        # Update logging variables
        with torch.no_grad():
            self.current_ratio = consensus_mask.mean().item()
            self.current_kappa = dynamic_threshold.item()

        return consensus_mask
