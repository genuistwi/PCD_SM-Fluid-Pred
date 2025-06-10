import numpy as np
from torch import nn
import lightning as lt

from diffusers import UNet2DConditionModel


class ImageToTokens(lt.LightningModule):

    def __init__(self, in_channels, cross_attention_dim):
        super(ImageToTokens, self).__init__()
        self.projection = nn.Linear(in_channels, cross_attention_dim)

    def forward(self, cond_image):
        """ cond_image shape: [batch_size, in_channels, height, width]. """
        b, c, h, w = cond_image.shape
        # Flatten spatially => [batch_size, c, h*w]
        flattened = cond_image.view(b, c, -1)  # shape [b, c, h*w]
        # Transpose => [batch_size, h*w, c]
        flattened = flattened.permute(0, 2, 1).contiguous()  # shape [b, h*w, c]
        # Now project from dimension c -> cross_attention_dim
        tokens = self.projection(flattened)  # shape [b, h*w, cross_attention_dim]
        return tokens


class C_UNet2DConditionModel(lt.LightningModule):
    def __init__(self, in_channels, condition_channels, model_channels):
        super(C_UNet2DConditionModel, self).__init__()

        cross_attention_dim = 256 // 4
        block_out = np.array([128, 256, 512, 512]) // 4

        self.unet = UNet2DConditionModel(
            sample_size=model_channels,
            in_channels=in_channels,             # e.g. if your main image is RGB
            out_channels=in_channels,
            cross_attention_dim=cross_attention_dim,   # dimension of your condition embeddings
            block_out_channels=tuple(block_out),  # 4 because 4 blocks below
            layers_per_block=2,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "CrossAttnUpBlock2D",
            ),
        )

        self.encoder = ImageToTokens(in_channels=condition_channels, cross_attention_dim=cross_attention_dim)

    def forward(self, x, t, cond):
        cond_emb = self.encoder(cond)
        t = t.squeeze()
        return self.unet(x, timestep=t, encoder_hidden_states=cond_emb).sample
