from __future__ import annotations

from typing import Dict, Optional, Sequence, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_utils import symmetric_binary_kl, binary_js_divergence, binary_kl_divergence, js_divergence

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        include_background: bool = False,
        class_weights: Optional[Sequence[float]] = None,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.include_background = include_background
        self.use_sigmoid = use_sigmoid
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else torch.empty(0, dtype=torch.float32)
        )
        self.register_buffer("class_weights", weight_tensor, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)
            
        probs = probs.flatten(2)
        targets = targets.flatten(2)
        
        intersection = (probs * targets).sum(-1)
        denominator = probs.sum(-1) + targets.sum(-1)
        
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        
        if self.class_weights.numel() > 0:
            weights = self.class_weights
            if weights.numel() != dice.shape[1]:
                 if weights.numel() > dice.shape[1]:
                     weights = weights[1:] 
            if weights.numel() == dice.shape[1]:
                weights = weights.to(dice.device)
                dice = dice * weights
            
        return 1 - dice.mean()


class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.weight is not None:
             w = self.weight.to(logits.device).view(1, -1, 1, 1, 1)
             focal_loss = focal_loss * w
        return focal_loss.mean()


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Optional[nn.Module] = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1e-5,
        ddp: bool = True,
    ):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor = None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            pass

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth
        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        return 1 - dc.mean()


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))
        if net_output.shape == gt.shape:
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=lambda x: torch.softmax(x, dim=1), **soft_dice_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            ce_loss = self.ce(net_output, target[:, 0].long())
        else:
            ce_loss = self.ce(net_output, target[:, 0].long())
        dc_loss = self.dc(net_output, target, loss_mask=None)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


# [Deleted TverskyLoss class from here]


class CombinedLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        pixel_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        dice_class_weights: Optional[Sequence[float]] = None,
        ce_label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, 
            {}, 
            weight_ce=pixel_weight, 
            weight_dice=dice_weight
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)
    

def compute_self_distillation_loss(
    seg_loss: nn.Module,
    main_logits: torch.Tensor,
    decoder_logits: Sequence[torch.Tensor],
    targets: torch.Tensor | List[torch.Tensor],
    decoder_weights: Sequence[float] | torch.Tensor | None,
    decoder_kl_weight: float,
    temperature: float,
    encoder_logits: Sequence[torch.Tensor] | None = None,
    encoder_weights: Sequence[float] | torch.Tensor | None = None,
    encoder_kl_weight: float = 0.0,
    clg_generator: nn.Module | None = None,
    clg_loss_weight: float = 1.0,
    include_background: bool = True,
    use_sigmoid: bool = True,
    use_js_div: bool = False,
    decoder_logits_upsampled: Sequence[torch.Tensor] | None = None,
    encoder_gt_enabled: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        spatial_dims = reference.shape[2:]
        if tensor.shape[2:] != spatial_dims:
            mode = "trilinear" if len(spatial_dims) == 3 else "bilinear"
            tensor = F.interpolate(tensor, size=spatial_dims, mode=mode, align_corners=False)
        return tensor

    def _prepare_weights(
        weights: Sequence[float] | torch.Tensor | None,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if count == 0:
            return torch.zeros(0, device=device, dtype=dtype)
        if weights is None:
            return torch.ones(count, device=device, dtype=dtype)
        if isinstance(weights, torch.Tensor):
            if weights.numel() != count:
                raise ValueError(f"Expected {count} weights, got {weights.numel()}.")
            return weights.to(device=device, dtype=dtype)
        tensor = torch.tensor(weights, device=device, dtype=dtype)
        if tensor.numel() != count:
            raise ValueError(f"Expected {count} weights, got {tensor.numel()}.")
        return tensor

    def _get_probs(logits: torch.Tensor, temp: float, use_sig: bool, include_bg: bool) -> torch.Tensor:
        # Clamp logits strictly to avoid inf/nan in exponentials
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        if use_sig:
            return torch.sigmoid(logits / temp)
        else:
            probs_full = torch.softmax(logits / temp, dim=1)
            if not include_bg and probs_full.shape[1] > 1:
                return probs_full[:, 1:, ...]
            return probs_full

    device = main_logits.device
    dtype = main_logits.dtype

    if isinstance(targets, list):
        target_main = targets[0]
    else:
        target_main = targets

    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    if isinstance(seg_loss, DeepSupervisionWrapper):
        total_loss = seg_loss([main_logits], [target_main])
    else:
        total_loss = seg_loss(main_logits, target_main)

    components: Dict[str, torch.Tensor] = {"seg_main": total_loss.detach()}

    decoder_weights_tensor = _prepare_weights(decoder_weights, len(decoder_logits), device, dtype)

    # --- 1. Decoder Deep Supervision ---
    seg_aux_total = torch.zeros_like(total_loss)
    if decoder_logits:
        for i, (weight, aux) in enumerate(zip(decoder_weights_tensor, decoder_logits)):
            if isinstance(targets, list):
                if i + 1 < len(targets):
                    target_aux = targets[i + 1]
                else:
                    target_aux = targets[-1]
            else:
                target_aux = target_main
            
            if aux.shape[2:] != target_aux.shape[2:]:
                aux = _resize_like(aux, target_aux)
            
            # Use safe clamping for aux logits before loss if necessary, but seg_loss (BCE/Dice) usually handles it.
            # However, ensure no NANs come from the model output directly.
            if torch.isnan(aux).any():
                aux = torch.nan_to_num(aux, nan=0.0)

            aux_seg = seg_loss(aux, target_aux)
            if torch.isnan(aux_seg):
                 aux_seg = torch.tensor(0.0, device=aux_seg.device, requires_grad=True)
            seg_aux_total = seg_aux_total + weight * aux_seg
            total_loss = total_loss + weight * aux_seg
    
    # --- 2. [Mod] Encoder Deep Supervision (GT Supervision) ---
    # User requested: Disable GT supervision for Encoder branch
    # But enabled during warmup to prevent dead gradients
    if encoder_logits and encoder_gt_enabled:
        # Reuse encoder_weights to give GT supervision signal to Encoder early
        encoder_gt_weights_tensor = _prepare_weights(encoder_weights, len(encoder_logits), device, dtype)
        
        for i, (weight, enc_logit) in enumerate(zip(encoder_gt_weights_tensor, encoder_logits)):
            if isinstance(targets, list):
                # For simplicity, use target_main (Highest Res) resized down, or deep supervision targets
                if i < len(targets): 
                    # Note: targets list is typically [Highest, Low, Lower...]
                    # encoder_logits is [Low, Higher, Highest...] (UNet Bottom-up)
                    # Needs corresponding matching.
                    # For simplicity, always resize High Res GT to current resolution
                    target_for_loss = target_main
                else:
                    target_for_loss = target_main
            else:
                target_for_loss = target_main

            if enc_logit.shape[2:] != target_for_loss.shape[2:]:
                enc_logit_aligned = _resize_like(enc_logit, target_for_loss)
            else:
                enc_logit_aligned = enc_logit

            # NaN Protection
            if torch.isnan(enc_logit_aligned).any():
                enc_logit_aligned = torch.nan_to_num(enc_logit_aligned, nan=0.0)

            loss_gt = seg_loss(enc_logit_aligned, target_for_loss)
            
            if torch.isnan(loss_gt):
                 loss_gt = torch.tensor(0.0, device=loss_gt.device, requires_grad=True)

            total_loss = total_loss + weight * loss_gt
            seg_aux_total = seg_aux_total + weight * loss_gt

    components["seg_aux"] = seg_aux_total.detach()

    # --- 3. Decoder Distillation ---
    decoder_kl_total = torch.zeros_like(total_loss)
    if (decoder_logits or decoder_logits_upsampled) and decoder_kl_weight > 0:
        teacher_base = main_logits.detach()
        teacher_probs = _get_probs(teacher_base, temperature, use_sigmoid, include_background)

        # Prefer upsampled logits if available, otherwise use raw logits
        student_logits = list(decoder_logits_upsampled) if decoder_logits_upsampled is not None else list(decoder_logits)
        student_weights = decoder_weights_tensor[:len(student_logits)]

        for weight, aux in zip(student_weights, student_logits):
            # Upsample student (Aux) to align with teacher (Main)
            aux_aligned = _resize_like(aux, main_logits)
            
            # [Mod] Use KL / JS for all distillation (replacing MSE)
            # User request: Always use KL for distillation, regardless of mode (Softmax/Sigmoid)
            if use_sigmoid:
                student_probs = _get_probs(aux_aligned, temperature, use_sigmoid, include_background)
                # Compute Binary KL Divergence: KL(Teacher || Student) to align with Softmax behavior (Reverse KL)
                # This ensures Student covers Teacher's mass.
                kl_map = binary_kl_divergence(teacher_probs, student_probs)
                if torch.isnan(kl_map).any():
                     kl_map = torch.nan_to_num(kl_map, nan=0.0)
                kl_val = kl_map.mean()
            else:
                # Softmax KL (Native PyTorch)
                with torch.autocast('cuda', enabled=False):
                    aux_aligned_f32 = aux_aligned.float()
                    teacher_probs_f32 = teacher_probs.float()
                    student_probs = _get_probs(aux_aligned_f32, temperature, use_sigmoid, include_background)
                    student_log_probs = torch.log(student_probs.clamp_min(1e-8))
                    kl_map = F.kl_div(student_log_probs, teacher_probs_f32, reduction="none")
                    kl_val = kl_map.sum(dim=1).mean()
            
            decoder_kl_total = decoder_kl_total + weight * kl_val
        total_loss = total_loss + decoder_kl_weight * (temperature ** 2) * decoder_kl_total
    components["kl_decoder"] = decoder_kl_total.detach() * (temperature ** 2) * decoder_kl_weight

    # --- 4. Encoder Distillation ---
    encoder_kl_total = torch.zeros_like(total_loss)
    mask_ratios = []
    if encoder_logits and encoder_kl_weight > 0:
        # Teacher T0 (Encoder Last) - Low Res
        teacher_enc_T0 = encoder_logits[-1].detach()
        # Teacher T1 (Decoder Main) - High Res
        teacher_dec_T1 = main_logits.detach()
        
        student_list = encoder_logits 
        encoder_weights_tensor = _prepare_weights(encoder_weights, len(student_list), device, dtype)
        
        for i, enc in enumerate(student_list):
            weight = encoder_weights_tensor[i]
            
            # Compute KL uniformly at Main Logits resolution (upsample everything)
            enc_aligned = _resize_like(enc, main_logits)
            student_probs = _get_probs(enc_aligned, temperature, use_sigmoid, include_background)
            
            # Prepare Teachers (also upsampled to Main)
            probs_T0 = _get_probs(_resize_like(teacher_enc_T0, main_logits), temperature, use_sigmoid, include_background)
            probs_T1 = _get_probs(_resize_like(teacher_dec_T1, main_logits), temperature, use_sigmoid, include_background)
            
            current_layer_loss = torch.zeros_like(total_loss)
            is_last_encoder = (i == len(student_list) - 1)
            
            # [A] Internal Distillation
            if not is_last_encoder:
                if use_sigmoid:
                    # KL(T0 || Student) = KL(Teacher || Student)
                    kl_internal = binary_kl_divergence(probs_T0, student_probs).mean()
                    if torch.isnan(kl_internal):
                         kl_internal = torch.tensor(0.0, device=kl_internal.device, requires_grad=True)
                else:
                    with torch.autocast('cuda', enabled=False):
                            student_probs_f32 = student_probs.float()
                            probs_T0_f32 = probs_T0.float()
                            # Use Softmax KL Divergence
                            # KL(Teacher || Student)
                            # input=log(student), target=teacher
                            student_log_probs = torch.log(student_probs_f32.clamp(1e-7, 1.0))
                            kl_internal = F.kl_div(student_log_probs, probs_T0_f32, reduction="none").sum(dim=1).mean()

            # [B] External Distillation (CLG)
            if is_last_encoder and clg_generator is not None:
                # Prepare aligned T0 and T1 for CLG
                t0_aligned = _resize_like(teacher_enc_T0, main_logits)
                t1_aligned = _resize_like(teacher_dec_T1, main_logits)
                mask_clg = clg_generator(t1_aligned, t0_aligned)

                if mask_clg is not None:
                    mask_ratios.append(mask_clg.mean())
                    if use_sigmoid:
                        # Use Binary JS for CLG Loss
                        clg_loss_val = binary_js_divergence(student_probs, probs_T1)
                        if torch.isnan(clg_loss_val).any():
                             clg_loss_val = torch.nan_to_num(clg_loss_val, nan=0.0)
                        
                        loss_external = (clg_loss_val * mask_clg).sum() / (mask_clg.sum() + 1e-6)
                    else:
                        with torch.autocast('cuda', enabled=False):
                            # Use JS Divergence for CLG
                            student_probs_f32 = student_probs.float()
                            probs_T1_f32 = probs_T1.float()
                            js_val = js_divergence(student_probs_f32, probs_T1_f32)
                            js_val = js_val.squeeze(1)
                            if torch.isnan(js_val).any():
                                js_val = torch.nan_to_num(js_val, nan=0.0)
                        
                        loss_external = (js_val * mask_clg).sum() / (mask_clg.sum() + 1e-6)
                    current_layer_loss = current_layer_loss + clg_loss_weight * loss_external

            encoder_kl_total = encoder_kl_total + weight * current_layer_loss

        total_loss = total_loss + encoder_kl_weight * (temperature ** 2) * encoder_kl_total

    components["kl_encoder"] = encoder_kl_total.detach() * (temperature ** 2) * encoder_kl_weight
    
    if mask_ratios:
        components["clg_mask_ratio"] = torch.stack(mask_ratios).mean()
    else:
        components["clg_mask_ratio"] = torch.tensor(0.0, device=device)

    return total_loss, components
