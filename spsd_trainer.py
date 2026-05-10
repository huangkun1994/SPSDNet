from __future__ import annotations
from typing import Union, Tuple, List

import torch
import numpy as np
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.network_initialization import InitWeights_He
from .spsd_unet import SPSD_UNet
from .distillation import DistillationController
import os

class SPSDTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
        self.initial_lr = 1e-3
        self.distillation_controller = None
        
        # Configurable SPSD weights for ablation studies
        # Parse from environment variables, or use default
        self.spsd_decoder_kl_weight = float(os.environ.get('SPSD_DECODER_KL_WEIGHT', 1.0))
        self.spsd_encoder_kl_weight = float(os.environ.get('SPSD_ENCODER_KL_WEIGHT', 1.0))
        
        # New: Parse Temperature and TopK Start from environment
        self.spsd_temp = float(os.environ.get('SPSD_TEMP', 1.5))
        self.spsd_k_start = float(os.environ.get('SPSD_K_START', 0.5))
        
        # Modify output folder to include ablation parameters
        # Format: SPSDTrainer_T[temp]_K[k_start]__...
        if 'SPSD_TEMP' in os.environ or 'SPSD_K_START' in os.environ:
             # Only modify if explicitly setting vars for ablation
             # This keeps standard runs neat, though technically same values
             new_folder_name = f'SPSDTrainer_T{self.spsd_temp}_K{self.spsd_k_start}__{plans["plans_name"]}__{configuration}'
             # output_folder_base is like .../nnUNet_results/DatasetXXX/
             # nnUNetTrainer calculates self.output_folder as join(folder_base, trainer_name__plans_name__config, fold_X)
             # We override it here.
             
             # Note: nnUNet trainer uses self.output_folder_base which is .../DatasetXXX/
             # The original self.output_folder was already set in super().__init__
             
             # Let's reconstruct it carefully
             self.output_folder = join(self.output_folder_base, new_folder_name, f'fold_{fold}')
             
             # Also update self.log_file path since it was likely set in super
             if hasattr(self, 'log_file'):
                 # It might not be set yet, usually set in run_training
                 pass
             print(f"SPSD Ablation: Modified output folder to {self.output_folder}")

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if hasattr(mod, '_orig_mod'):
            mod = mod._orig_mod
        if hasattr(mod, 'use_aux_outputs'):
            mod.use_aux_outputs = enabled
        if hasattr(mod, 'use_encoder_outputs'):
            mod.use_encoder_outputs = enabled

    def initialize(self):
        super().initialize()

        if self.is_ddp:
            self.network = self.network.module
            self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

        seg_loss = self.loss
        if isinstance(seg_loss, DeepSupervisionWrapper):
            seg_loss = seg_loss.loss

        use_sigmoid = self.label_manager.has_regions if hasattr(self, 'label_manager') else True
        
        if hasattr(self, 'configuration_manager'):
            depth = len(self.configuration_manager.network_arch_init_kwargs['kernel_sizes'])
            decoder_weights = [0.5 / (2**i) for i in range(depth)]
            # Encoder weights: stronger at shallow layers, weaker at deep layers
            encoder_weights = [1.0, 0.75, 0.5, 0.25, 0.125][:depth]
            if len(encoder_weights) < depth:
                encoder_weights = [0.5] * depth
        else:
            decoder_weights = [0.5, 0.25, 0.125, 0.0625]
            encoder_weights = [1.0, 0.75, 0.5, 0.25]

        self.distillation_controller = DistillationController(
            seg_loss=seg_loss,
            enabled=True,
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,

            decoder_kl_weight=self.spsd_decoder_kl_weight, 
            encoder_kl_weight=self.spsd_encoder_kl_weight, 
            temperature=self.spsd_temp,       

            kl_warmup_epochs=0,
            learnable_weights=False,
            clg_loss_weight=1.0, 
            clg_enabled=True,
            clg_method='topk', 
            clg_kappa_start=self.spsd_k_start, 
            clg_kappa_end=0.9999,  
            clg_threshold=0.1,   
            clg_sigma=0.5,
            clg_rampup_epochs=150,       
            total_epochs=self.num_epochs,
            decoder_start_epoch=0,
            encoder_start_epoch=0,
            use_sigmoid=use_sigmoid,
            use_js_div=True,
            encoder_gt_warmup_epochs=50 # Train encode with GT for first 50 epochs
        )

    def on_epoch_end(self):
        super().on_epoch_end()
        if (self.current_epoch + 1) % 10 == 0:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_epoch_{self.current_epoch + 1}.pth'))

    @staticmethod
    def build_network_architecture(architecture_class_name: str, arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        print(f"DEBUG SPSD_UNet Build: num_input_channels={num_input_channels}, num_output_channels={num_output_channels}")
        
        model = SPSD_UNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels, 
            base_channels=32, 
            depth=len(arch_init_kwargs['kernel_sizes']),
            dropout=0.0,
            use_aux_outputs=True, 
            use_encoder_outputs=True, 
        )
        model.apply(InitWeights_He(1e-2))
        
        if not enable_deep_supervision:
            model.use_aux_outputs = False
            if hasattr(model, 'use_encoder_outputs'):
                model.use_encoder_outputs = False
                
        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, return_features=False)
            l, logs = self.distillation_controller.compute(
                output, 
                target, 
                epoch=self.current_epoch
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy(), **logs}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data, return_features=False)
                l, logs = self.distillation_controller.compute(
                    output, 
                    target, 
                    epoch=self.current_epoch
                )

        if isinstance(output, dict):
            output_tensor = output['logits_main']
            encoder_logits_list = output.get('logits_encoder', [])
        else:
            output_tensor = output
            encoder_logits_list = []

        if isinstance(target, list):
            target_tensor = target[0]
        else:
            target_tensor = target

        axes = [0] + list(range(2, output_tensor.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_tensor) > 0.5).long()
        else:
            output_seg = output_tensor.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_tensor.shape, device=output_tensor.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_tensor != self.label_manager.ignore_label).float()
                target_tensor[target_tensor == self.label_manager.ignore_label] = 0
            else:
                if target_tensor.dtype == torch.bool:
                    mask = ~target_tensor[:, -1:]
                else:
                    mask = 1 - target_tensor[:, -1:]
                target_tensor = target_tensor[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_tensor, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        tp_enc, fp_enc, fn_enc = 0, 0, 0
        if len(encoder_logits_list) > 0:
            enc_last = encoder_logits_list[-1]
            if enc_last.shape[2:] != target_tensor.shape[2:]:
                enc_last = torch.nn.functional.interpolate(enc_last, size=target_tensor.shape[2:], mode='trilinear', align_corners=False)
            
            if self.label_manager.has_regions:
                pred_enc = (torch.sigmoid(enc_last) > 0.5).long()
            else:
                out_seg_enc = enc_last.argmax(1)[:, None]
                pred_enc = torch.zeros(enc_last.shape, device=enc_last.device, dtype=torch.float32)
                pred_enc.scatter_(1, out_seg_enc, 1)
            
            tpe, fpe, fne, _ = get_tp_fp_fn_tn(pred_enc, target_tensor, axes=axes, mask=mask)
            tp_enc = tpe.detach().cpu().numpy()
            fp_enc = fpe.detach().cpu().numpy()
            fn_enc = fne.detach().cpu().numpy()

            if not self.label_manager.has_regions:
                tp_enc = tp_enc[1:]
                fp_enc = fp_enc[1:]
                fn_enc = fn_enc[1:]

        return {
            'loss': l.detach().cpu().numpy(), 
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
            'tp_enc': tp_enc, 'fp_enc': fp_enc, 'fn_enc': fn_enc 
        }

class SPSDTrainer_EncoderOnly(SPSDTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Disable decoder KL, keep encoder KL
        self.spsd_decoder_kl_weight = 0.0
        self.spsd_encoder_kl_weight = 1.0


class SPSDTrainer_DecoderOnly(SPSDTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Disable encoder KL, keep decoder KL
        self.spsd_decoder_kl_weight = 1.0
        self.spsd_encoder_kl_weight = 0.0


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        tp_enc = [i.get('tp_enc', 0) for i in val_outputs]
        fp_enc = [i.get('fp_enc', 0) for i in val_outputs]
        fn_enc = [i.get('fn_enc', 0) for i in val_outputs]
        valid_indices = [i for i, val in enumerate(tp_enc) if not isinstance(val, int)]
        
        if len(valid_indices) > 0:
            tp_sum = np.sum([tp_enc[i] for i in valid_indices], axis=0)
            fp_sum = np.sum([fp_enc[i] for i in valid_indices], axis=0)
            fn_sum = np.sum([fn_enc[i] for i in valid_indices], axis=0)
            global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 
                                   for i, j, k in zip(tp_sum, fp_sum, fn_sum)]
            mean_dice_enc = np.mean(global_dc_per_class)
            self.print_to_log_file(f"Encoder (Last) Mean Dice: {mean_dice_enc:.4f}")

    # Moved methods to SPSDTrainer base class

    # def set_deep_supervision_enabled... (Already in base)
    
    # def train_step... (Already in base)

class SPSDTrainer_Threshold(SPSDTrainer):
    def initialize(self):
        super().initialize()
        use_sigmoid = self.distillation_controller.use_sigmoid
        decoder_weights = self.distillation_controller.decoder_weight_buffer.cpu().tolist()
        encoder_weights = self.distillation_controller.encoder_weight_buffer.cpu().tolist()

        seg_loss = self.loss
        if isinstance(seg_loss, DeepSupervisionWrapper):
            seg_loss = seg_loss.loss
        
        self.distillation_controller = DistillationController(
            seg_loss=seg_loss,
            enabled=True,
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            decoder_kl_weight=0.5,
            encoder_kl_weight=0.5,
            temperature=1.5,
            kl_warmup_epochs=50,
            learnable_weights=False,
            clg_loss_weight=0.5,
            clg_enabled=True,
            clg_method='threshold', 
            clg_threshold=0.1,    
            total_epochs=self.num_epochs,
            decoder_start_epoch=20,
            encoder_start_epoch=20,
            use_sigmoid=use_sigmoid,
            use_js_div=not use_sigmoid
        )

class SPSDTrainer_Dynamic(SPSDTrainer):
    def initialize(self):
        super().initialize()
        use_sigmoid = self.distillation_controller.use_sigmoid
        decoder_weights = self.distillation_controller.decoder_weight_buffer.cpu().tolist()
        encoder_weights = self.distillation_controller.encoder_weight_buffer.cpu().tolist()

        seg_loss = self.loss
        if isinstance(seg_loss, DeepSupervisionWrapper):
            seg_loss = seg_loss.loss
        
        self.distillation_controller = DistillationController(
            seg_loss=seg_loss,
            enabled=True,
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            decoder_kl_weight=0.5,
            encoder_kl_weight=0.5,
            temperature=1.5,
            kl_warmup_epochs=50,
            learnable_weights=False,
            clg_loss_weight=0.5,
            clg_enabled=True,
            clg_method='dynamic',
            clg_sigma=0.5,        
            total_epochs=self.num_epochs,
            decoder_start_epoch=20,
            encoder_start_epoch=20,
            use_sigmoid=use_sigmoid,
            use_js_div=not use_sigmoid
        )

class SPSDTrainer_NoCLG(SPSDTrainer):
    def initialize(self):
        super().initialize()
        use_sigmoid = self.distillation_controller.use_sigmoid
        decoder_weights = self.distillation_controller.decoder_weight_buffer.cpu().tolist()
        encoder_weights = self.distillation_controller.encoder_weight_buffer.cpu().tolist()

        seg_loss = self.loss
        if isinstance(seg_loss, DeepSupervisionWrapper):
            seg_loss = seg_loss.loss
        
        self.distillation_controller = DistillationController(
            seg_loss=seg_loss,
            enabled=True,
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            decoder_kl_weight=0.5,
            encoder_kl_weight=0.5,
            temperature=1.5,
            kl_warmup_epochs=50,
            learnable_weights=False,
            clg_loss_weight=0.0,
            clg_enabled=False,
            total_epochs=self.num_epochs,
            decoder_start_epoch=20,
            encoder_start_epoch=20,
            use_sigmoid=use_sigmoid,
            use_js_div=not use_sigmoid
        )