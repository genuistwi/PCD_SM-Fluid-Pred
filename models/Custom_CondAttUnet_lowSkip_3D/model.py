import lightning as lt

import torch
import torch.nn as nn
from diffusers import UNet3DConditionModel

torch.autograd.set_detect_anomaly(True)

class Conditional3DUNetWrapper(lt.LightningModule):
    def __init__(
            self,
            unet_in_channels: int = 4,
            unet_out_channels: int = 4,
            block_out_channels = (128, 256, 256),
            cross_attention_dim: int = 256,
            hidden_dim: int = 128,
            attention_head_dim=8,
            down_block_types=None,
            up_block_types=None,
            concat=True, # condition concatenated with input for conditioning
    ):
        """
        Args:
          unet_in_channels:  Number of input channels for the 3D UNet (e.g. physics fields).
          unet_out_channels: Number of output channels for the 3D UNet.
          block_out_channels:     List controlling the UNet’s channel counts.
          cross_attention_dim: Dimensionality of cross-attention conditioning.
          max_time:         Upper bound for timesteps (if using embedding).
          hidden_dim:       Hidden dim for MLP embeddings of time and noise level.
        """
        super().__init__()

        self.concat = concat

        # 1) Create the UNet3DConditionModel
        self.unet = UNet3DConditionModel(
            sample_size=None,  # can be None if you want arbitrary input sizes
            in_channels=2*unet_in_channels if concat else unet_in_channels,
            out_channels=unet_out_channels,
            # base_channels / down_block_types / up_block_types can be configured via the "block_out_channels" param
            block_out_channels=block_out_channels,
            cross_attention_dim=cross_attention_dim,
            # mid_block_type="UNetMidBlock3DCrossAttn",  # example mid-block that supports cross-attn
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            attention_head_dim=attention_head_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cross_attention_dim),
        )

        self.noise_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cross_attention_dim),
        )

        if not self.concat:
            self.cond_encoder = nn.Sequential(
                nn.Conv3d(
                    in_channels=unet_in_channels,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),  # produce [batch, 8, 1, 1, 1]
            )
            self.cond_proj = nn.Linear(8, cross_attention_dim)

        self.gate_head = nn.Sequential(
            nn.Linear(cross_attention_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor, noise_level: torch.Tensor):
        """
        Inputs:
          x:           [batch, in_channels, Lx, Ly, Lz]       main 3D input (noisy sample)
          time:        [batch, 1]                            scalar timestep
          cond:        [batch, in_channels, Lx, Ly, Lz]       condition 3D volume
          noise_level: [batch, 1]                            scalar noise level
        """

        time = time.float().clone()
        if len(time.shape) == 1:
            time = time.unsqueeze(-1)

        noise_level = noise_level.clone()

        # 1) Embed time and noise
        #    The shape of time/noise_level is [B,1], so we flatten or keep as is for the MLP
        time_emb = self.time_mlp(time)  # [B, cross_attention_dim]
        noise_emb = self.noise_mlp(noise_level)  # [B, cross_attention_dim]

        # Combine them (example: simple addition, or you could concatenate)
        combined_emb = time_emb + noise_emb  # [B, cross_attention_dim]

        if not self.concat:
            # 2) Encode condition (cond) → cross-attention
            cond_encoded = self.cond_encoder(cond)  # [B, 8, 1, 1, 1]
            cond_encoded = cond_encoded.view(cond.size(0), -1)  # [B, 8]

            cond_emb = self.cond_proj(cond_encoded)  # [B, cross_attention_dim]

            cross_attn_emb = cond_emb + combined_emb  # [B, cross_attention_dim]
            cross_attn_emb = cross_attn_emb.unsqueeze(1) # [B, 1, cross_attention_dim]

        time_squeezed = time.squeeze(-1).float().clone()

        # assuming x is the same shape as cond, input has twice as many channels
        conditioned_input = torch.cat([x, cond], dim=1)  # [B, C, ...]

        if not self.concat:
            unet_out = self.unet(
                sample=x,
                timestep=time_squeezed,  # If time is [B, 1], pass just [B]
                encoder_hidden_states=cross_attn_emb
            )
        else:
            combined_emb = combined_emb.unsqueeze(1)
            unet_out = self.unet(
                sample=conditioned_input,  # or just x
                timestep=time_squeezed,  # If time is [B, 1], pass just [B]
                encoder_hidden_states=combined_emb
            )

        # --- Gate ---
        alpha = self.gate_head(time_emb)  # shape [B,1]
        alpha = alpha[..., None, None, None]
        alpha = torch.clamp(alpha, min=0.01, max=0.99)
        out = x + alpha * unet_out.sample

        return  out
