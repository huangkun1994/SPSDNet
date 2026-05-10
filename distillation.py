from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from .losses import compute_self_distillation_loss
from .CLGblock import ConsensusMaskGenerator_TopK, ConsensusMaskGenerator_Threshold, ConsensusMaskGenerator_Dynamic


class DistillationController(nn.Module):
    def __init__(
        self,
        seg_loss: nn.Module,
        *,
        enabled: bool,
        decoder_weights: Sequence[float] | None,
        encoder_weights: Sequence[float] | None,
        decoder_kl_weight: float,
        encoder_kl_weight: float,
        temperature: float,
        kl_warmup_epochs: int = 0,
        learnable_weights: bool = False,
        clg_loss_weight: float = 1.0,
        clg_enabled: bool = True,
        clg_method: str = 'topk', 
        clg_kappa_start: float = 0.5,
        clg_kappa_end: float = 0.98,
        clg_threshold: float = 0.1,
        clg_sigma: float = 0.5,
        clg_rampup_epochs: Optional[int] = None, # Added parameter
        total_epochs: int = 200,
        decoder_start_epoch: int = 0,
        encoder_start_epoch: int = 0,
        decoder_kl_warmup_epochs: Optional[int] = None,
        encoder_kl_warmup_epochs: Optional[int] = None,
        use_sigmoid: bool = True,
        use_js_div: bool = False,
        encoder_gt_warmup_epochs: int = 50, # Added param
    ) -> None:
        super().__init__()
        self.seg_loss = seg_loss  
        self.encoder_gt_warmup_epochs = encoder_gt_warmup_epochs
        self.enabled = bool(enabled)
        self.learnable_weights = bool(learnable_weights)
        self.decoder_kl_base = float(decoder_kl_weight)
        self.encoder_kl_base = float(encoder_kl_weight)
        self.temperature = float(temperature)
        self.clg_loss_weight = float(clg_loss_weight)
        self.kl_warmup_epochs = max(0, int(kl_warmup_epochs))
        self.decoder_start_epoch = max(0, int(decoder_start_epoch))
        self.encoder_start_epoch = max(0, int(encoder_start_epoch))
        self.decoder_kl_warmup_epochs = (
            None if decoder_kl_warmup_epochs is None else max(0, int(decoder_kl_warmup_epochs))
        )
        self.encoder_kl_warmup_epochs = (
            None if encoder_kl_warmup_epochs is None else max(0, int(encoder_kl_warmup_epochs))
        )
        self.use_sigmoid = use_sigmoid
        self.use_js_div = use_js_div
        self.register_buffer(
            "decoder_weight_buffer",
            torch.tensor(decoder_weights or [], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "encoder_weight_buffer",
            torch.tensor(encoder_weights or [], dtype=torch.float32),
            persistent=False,
        )
        self.decoder_weight_param: Optional[nn.Parameter] = None
        self.encoder_weight_param: Optional[nn.Parameter] = None
        if self.learnable_weights:
            if self.decoder_weight_buffer.numel() > 0:
                self.decoder_weight_param = nn.Parameter(self.decoder_weight_buffer.clone())
                self.decoder_weight_buffer = torch.empty(0, dtype=torch.float32)
            if self.encoder_weight_buffer.numel() > 0:
                self.encoder_weight_param = nn.Parameter(self.encoder_weight_buffer.clone())
                self.encoder_weight_buffer = torch.empty(0, dtype=torch.float32)
        self.clg_enabled = clg_enabled
        self.clg_method = clg_method
        self.clg_threshold = clg_threshold
        self.clg_sigma = clg_sigma

        if self.enabled and self.clg_enabled:
            if self.clg_method == 'topk':
                rampup = clg_rampup_epochs if clg_rampup_epochs is not None else total_epochs
                self.clg_generator = ConsensusMaskGenerator_TopK(
                    kappa_start=clg_kappa_start, 
                    kappa_end=clg_kappa_end, 
                    rampup_epochs=rampup, 
                    use_sigmoid=use_sigmoid
                )
            elif self.clg_method == 'threshold':
                self.clg_generator = ConsensusMaskGenerator_Threshold(
                    threshold=self.clg_threshold,
                    use_sigmoid=use_sigmoid
                )
            elif self.clg_method == 'dynamic':
                self.clg_generator = ConsensusMaskGenerator_Dynamic(
                    sigma=self.clg_sigma,
                    use_sigmoid=use_sigmoid
                )
            else:
                raise ValueError(f"Unknown clg_method: {self.clg_method}. Choose from 'topk', 'threshold', 'dynamic'")
        else:
            self.clg_generator = None


    def _scheduled_weight(self, *, base: float, epoch: int, start_epoch: int, warmup_epochs: Optional[int]) -> float:
        if not self.enabled or base <= 0:
            return 0.0
        if epoch < start_epoch:
            return 0.0
        wu = self.kl_warmup_epochs if warmup_epochs is None else warmup_epochs
        if wu <= 0:
            return base
        steps = max(0, epoch - start_epoch + 1)
        factor = min(1.0, float(steps) / float(wu))
        return base * factor


    def _resolve_weights(self, kind: str, count: int, device: torch.device) -> torch.Tensor:
        if count == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)
        param: Optional[nn.Parameter] = getattr(self, f"{kind}_weight_param")
        buffer: torch.Tensor = getattr(self, f"{kind}_weight_buffer")
        if param is not None:
            if param.numel() < count:
                raise ValueError(f"{kind} weight parameter length {param.numel()} < required {count}")
            raw = param[:count]
            if raw.device != device:
                raise RuntimeError(f"{kind} weight parameter device {raw.device} does not match target {device}")
            return torch.softmax(raw, dim=0)
        if buffer.numel() == 0:
            return torch.ones(count, dtype=torch.float32, device=device)
        if buffer.numel() < count:
            raise ValueError(f"{kind} weight buffer length {buffer.numel()} < required {count}")
        return buffer[:count].to(device=device)
    

    def compute(
        self,
        outputs,
        targets: torch.Tensor | List[torch.Tensor],
        *,
        epoch: int,
        teacher_main_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.enabled or not isinstance(outputs, dict) or "logits_main" not in outputs:
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.get("logits_main")
            target_main = targets[0] if isinstance(targets, list) else targets
            total = self.seg_loss(logits, target_main)
            return total, {
                "seg_main": float(total.detach().item()),
                "seg_aux": 0.0,
                "kl_decoder": 0.0,
                "kl_encoder": 0.0,
                "kl_decoder_weight": 0.0,
                "kl_encoder_weight": 0.0,
            }

        main_logits: torch.Tensor = outputs["logits_main"]
        decoder_logits: List[torch.Tensor] = list(outputs.get("logits_decoder", []))
        # decoder_logits_upsampled list may be empty as we removed the upsampling step
        decoder_logits_upsampled: List[torch.Tensor] = list(outputs.get("logits_decoder_upsampled", []))
        encoder_logits: List[torch.Tensor] = list(outputs.get("logits_encoder", []))

        decoder_w = self._scheduled_weight(
            base=self.decoder_kl_base,
            epoch=epoch,
            start_epoch=self.decoder_start_epoch,
            warmup_epochs=self.decoder_kl_warmup_epochs,
        )
        encoder_w = self._scheduled_weight(
            base=self.encoder_kl_base,
            epoch=epoch,
            start_epoch=self.encoder_start_epoch,
            warmup_epochs=self.encoder_kl_warmup_epochs,
        )

        decoder_weight_tensor = self._resolve_weights("decoder", len(decoder_logits), main_logits.device)
        encoder_weight_tensor = self._resolve_weights("encoder", len(encoder_logits), main_logits.device)

        if self.clg_generator is not None:
            self.clg_generator.update_kappa(epoch)

        # Check Warmup for Encoder GT
        encoder_gt_enabled = False
        if hasattr(self, 'encoder_gt_warmup_epochs'):
            encoder_gt_enabled = epoch < self.encoder_gt_warmup_epochs

        total, parts = compute_self_distillation_loss(
            seg_loss=self.seg_loss,
            main_logits=main_logits,
            decoder_logits=decoder_logits,
            targets=targets,
            decoder_weights=decoder_weight_tensor,
            decoder_kl_weight=decoder_w,
            temperature=self.temperature,
            encoder_logits=encoder_logits,
            encoder_weights=encoder_weight_tensor,
            encoder_kl_weight=encoder_w,
            clg_generator=self.clg_generator,
            clg_loss_weight=self.clg_loss_weight,
            use_sigmoid=self.use_sigmoid,
            use_js_div=self.use_js_div,
            decoder_logits_upsampled=decoder_logits_upsampled,
            encoder_gt_enabled=encoder_gt_enabled,
        )
        
        zero = torch.zeros((), device=main_logits.device)
        logs = {
            "seg_main": float(parts.get("seg_main", zero).item()),
            "seg_aux": float(parts.get("seg_aux", zero).item()),
            "kl_decoder": float(parts.get("kl_decoder", zero).item()),
            "kl_encoder": float(parts.get("kl_encoder", zero).item()),
            "kl_decoder_weight": decoder_w,
            "kl_encoder_weight": encoder_w,
            "clg_mask_ratio": float(parts.get("clg_mask_ratio", zero).item()),
        }
        if self.learnable_weights:
            if decoder_weight_tensor.numel() > 0:
                logs["decoder_weights"] = decoder_weight_tensor.detach().cpu().tolist()
            if encoder_weight_tensor.numel() > 0:
                logs["encoder_weights"] = encoder_weight_tensor.detach().cpu().tolist()
        if self.clg_generator is not None:
            logs["clg_kappa"] = self.clg_generator.current_kappa
        return total, logs
