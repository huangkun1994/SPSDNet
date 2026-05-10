import torch
import torch.nn.functional as F
from math import exp


def _check_and_align_shapes(t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function: checks and aligns spatial dimensions.
    Uses trilinear interpolation to resize to the larger dimension if sizes mismatch.
    """
    if t1.shape[2:] != t2.shape[2:]:
        target_size = tuple([max(s1, s2) for s1, s2 in zip(t1.shape[2:], t2.shape[2:])])
        if t1.shape[2:] != target_size:
            t1 = F.interpolate(t1, size=target_size, mode="trilinear", align_corners=False)
        if t2.shape[2:] != target_size:
            t2 = F.interpolate(t2, size=target_size, mode="trilinear", align_corners=False)
    return t1, t2


def js_divergence(p1_softmax: torch.Tensor, p2_softmax: torch.Tensor) -> torch.Tensor:
    p1_softmax = p1_softmax.clamp(min=1e-7)
    p2_softmax = p2_softmax.clamp(min=1e-7)

    if p1_softmax.shape[2:] != p2_softmax.shape[2:]:#Align spatial dimensions
        target_size = [max(s1, s2) for s1, s2 in zip(p1_softmax.shape[2:], p2_softmax.shape[2:])]
        if p1_softmax.shape[2:] != tuple(target_size):
            p1_softmax = F.interpolate(p1_softmax, size=target_size, mode="trilinear", align_corners=False)
        if p2_softmax.shape[2:] != tuple(target_size):
            p2_softmax = F.interpolate(p2_softmax, size=target_size, mode="trilinear", align_corners=False)

    m = 0.5 * (p1_softmax + p2_softmax)
    m = m.clamp(min=1e-7)
    log_p1 = torch.log(p1_softmax)
    log_p2 = torch.log(p2_softmax)
    
    # KL(P||M) = sum(P * log(P/M)) = sum(P * (logP - logM))
    # F.kl_div(input, target) = target * (log(target) - input)
    # So input=logM, target=P
    log_m = torch.log(m)
    kl1 = F.kl_div(log_m, p1_softmax, reduction="none").sum(dim=1, keepdim=True)
    kl2 = F.kl_div(log_m, p2_softmax, reduction="none").sum(dim=1, keepdim=True)

    return 0.5 * (kl1 + kl2)      


def binary_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculate Binary KL Divergence KL(P || Q) using PyTorch's BCE for stability.
    KL(p||q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
             = BCE(q, p) - BCE(p, p)
    where BCE(input=q, target=p) = - [p * log(q) + (1-p) * log(1-q)]
    """
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    
    # BCE(input=q, target=p)
    # Note: F.binary_cross_entropy requires input (q) to be probabilities [0,1].
    # p is target.
    try:
        bce_q_p = F.binary_cross_entropy(q, p, reduction='none')
        bce_p_p = F.binary_cross_entropy(p, p, reduction='none')
    except (RuntimeError, ValueError) as e:
        # Fallback to manual if shape mismatch or other error, though shape should be aligned
        # Revert to safe manual calculation
        term1 = p * (torch.log(p) - torch.log(q))
        term2 = (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))
        return term1 + term2

    return bce_q_p - bce_p_p

def binary_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculate Binary Jensen-Shannon Divergence.
    JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), where m = 0.5 * (p + q)
    Bound: [0, ln(2)] ~= [0, 0.693]
    """
    p, q = _check_and_align_shapes(p, q)
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    m = 0.5 * (p + q)
    
    kl_p_m = binary_kl_divergence(p, m, eps)
    kl_q_m = binary_kl_divergence(q, m, eps)
    
    return 0.5 * (kl_p_m + kl_q_m)

def symmetric_binary_kl(p, q, eps=1e-7):
    # This function is kept for backward compatibility if needed, 
    # but the user requested to use JS divergence.
    # We might redirect this or just keep it as is and change the call site.
    p, q = _check_and_align_shapes(p, q)
    kl_p_q = binary_kl_divergence(p, q, eps)
    kl_q_p = binary_kl_divergence(q, p, eps)
    raw_kl = 0.5 * (kl_p_q + kl_q_p)
    
    return raw_kl.mean(dim=1, keepdim=True)


def _gaussian_window_3d(window_size: int, sigma: float, channel: int):
    _1d_window = torch.tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    _1d_window = _1d_window / _1d_window.sum()
    
    # Generate 3D window via outer product
    _1d_window_u = _1d_window.unsqueeze(1)
    _2d_window = _1d_window_u.mm(_1d_window_u.t())
    _3d_window = _2d_window.unsqueeze(2) @ _1d_window.unsqueeze(0) # (W, W, W)
    
    window = _3d_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def ssim_3d_map(p1: torch.Tensor, p2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    p1, p2 = _check_and_align_shapes(p1, p2)
    channel = p1.size(1)
    window = _gaussian_window_3d(window_size, sigma, channel).to(p1.device).type_as(p1)
    
    mu1 = F.conv3d(p1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(p2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(p1 * p1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(p2 * p2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(p1 * p2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    return ssim_map.mean(dim=1, keepdim=True)


def mse_loss_map(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    p1, p2 = _check_and_align_shapes(p1, p2)
    # Compute squared differences
    diff_sq = (p1 - p2) ** 2
    
    # Average across channel dimension
    mse_map = diff_sq.mean(dim=1, keepdim=True)
    return mse_map
