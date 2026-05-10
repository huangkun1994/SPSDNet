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
        
        

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if hasattr(mod, '_orig_mod'):
            mod = mod._orig_mod
        if hasattr(mod, 'use_aux_outputs'):
            mod.use_aux_outputs = enabled
        

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
            encoder_weights = [1.0, 0.75, 0.5, 0.25, 0.125][:depth]
            if len(encoder_weights) < depth:
                encoder_weights.extend([0.5] * (depth - len(encoder_weights)))
        else:
            decoder_weights = [0.5, 0.25, 0.125, 0.0625]
            encoder_weights = [1.0, 0.75, 0.5, 0.25]

        self.distillation_controller = DistillationController(
            seg_loss=seg_loss,
            enabled=True,
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            decoder_kl_weight=1.0,
            encoder_kl_weight=1.0,
            temperature=1.5,
            kl_warmup_epochs=0,
            learnable_weights=False,
            dcg_loss_weight=1.0,
            dcg_enabled=True,
            dcg_method='topk',
            dcg_kappa_start=0.5,
            dcg_kappa_end=0.9999,
            dcg_threshold=0.1,
            dcg_sigma=0.5,
            dcg_rampup_epochs=150,
            total_epochs=self.num_epochs,
            decoder_start_epoch=0,
            encoder_start_epoch=0,
            use_sigmoid=use_sigmoid,
            use_js_div=True
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
            pass
        else:
            output_tensor = output
            pass

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

        

        return {
            'loss': l.detach().cpu().numpy(), 
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
             
        }

