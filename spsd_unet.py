from __future__ import annotations

from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        def _norm(c: int) -> nn.Module:
            return nn.InstanceNorm3d(c, affine=True)
        
        def _act() -> nn.Module:
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)

        layers: List[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),  
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),  
            _act(),               
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv_block = ConvBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.norm(x)
        x = self.act(x)
        return self.conv_block(x)
    

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd shapes
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
    

class SPSD_UNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        use_aux_outputs: bool = False,
        use_encoder_outputs: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        if 'n_stages' in kwargs:
            depth = kwargs['n_stages']
            
        assert depth >= 1, "UNet depth must be at least 1"
        self.inc = ConvBlock(input_channels, base_channels, dropout)
        self.last_feature: torch.Tensor | None = None
        self.use_aux_outputs = use_aux_outputs
        self.use_encoder_outputs = use_encoder_outputs

        downs: List[nn.Module] = []
        ups: List[nn.Module] = []
        aux_heads: List[nn.Module] = []
        encoder_heads: List[nn.Module] = []

        channels = base_channels
        skip_channels: List[int] = []
        encoder_channels: List[int] = [base_channels]
        
        # Build Encoder
        for _ in range(depth - 1):
            next_channels = channels * 2
            downs.append(DownBlock(channels, next_channels, dropout))
            skip_channels.append(channels)
            channels = next_channels
            encoder_channels.append(channels)

        self.downs = nn.ModuleList(downs)
        bottleneck_channels = channels * 2
        self.bottleneck = ConvBlock(channels, bottleneck_channels, dropout)
        channels = bottleneck_channels

        # Build Decoder
        for level, skip in enumerate(reversed(skip_channels)):
            ups.append(UpBlock(channels, skip, dropout))
            if use_aux_outputs:
                # Decoder Aux Heads: Standard 1x1 Conv
                aux_heads.append(nn.Conv3d(skip, num_classes, kernel_size=1))
            channels = skip

        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv3d(channels, num_classes, kernel_size=1)
        self.aux_heads = nn.ModuleList(aux_heads) if use_aux_outputs else None
        
        # [Added] Bottleneck Head for Deepest Supervision
        if use_aux_outputs:
            self.bottleneck_head = nn.Conv3d(bottleneck_channels, num_classes, kernel_size=1)
        else:
            self.bottleneck_head = None

        # Build Encoder Heads: Standard 1x1 Conv (Low Resolution Output)
        if self.use_encoder_outputs:
            for ch in encoder_channels:
                encoder_heads.append(nn.Conv3d(ch, num_classes, kernel_size=1))
            self.encoder_heads = nn.ModuleList(encoder_heads)
        else:
            self.encoder_heads = None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        encoder_features: List[torch.Tensor] = []
        decoder_features: List[torch.Tensor] = []

        # Encoder Path
        x_curr = self.inc(x)
        encoder_features.append(x_curr)
        for down in self.downs:
            x_curr = down(x_curr)
            encoder_features.append(x_curr)

        x_curr = self.bottleneck(x_curr)

        # Decoder Path
        aux_outputs: List[torch.Tensor] = []
        
        # [Added] Bottleneck Output (Deepest)
        if self.use_aux_outputs and self.bottleneck_head is not None:
             aux_outputs.append(self.bottleneck_head(x_curr))

        for idx, (up, skip) in enumerate(zip(self.ups, reversed(encoder_features[:-1]))):
            x_curr = up(x_curr, skip)
            decoder_features.append(x_curr)
            if self.use_aux_outputs and self.aux_heads is not None:
                # Outputs low-resolution logits
                aux_outputs.append(self.aux_heads[idx](x_curr))

        self.last_feature = x_curr
        logits = self.outc(x_curr) # Main Output

        has_aux = self.use_aux_outputs and self.aux_heads is not None
        has_encoder = self.use_encoder_outputs and self.encoder_heads is not None
        if not has_aux and not has_encoder:
            return logits

        payload = {
            "logits_main": logits,
        }

        if has_aux:
            # Reverse to be [Shallow, ..., Deep] for nnU-Net convention
            payload["logits_decoder"] = aux_outputs[::-1] 

        if has_encoder:
            encoder_logits: List[torch.Tensor] = []
            if self.encoder_heads is not None:
                for head, feature in zip(self.encoder_heads, encoder_features):
                    encoder_logits.append(head(feature))
            payload["logits_encoder"] = encoder_logits

        return payload

    def compute_conv_feature_map_size(self, input_size):
        return None