from models.Custom_CondAttUnet_lowSkip_3D.model import Conditional3DUNetWrapper
import torch
import numpy as np

def model_setup(Cfg, data_set):
    """
    3D adaptation
    """

    assert data_set.dataset_name in ["MHD_64"]

    div = 10

    model = Conditional3DUNetWrapper(
        unet_in_channels=data_set.fields_len,
        unet_out_channels=data_set.fields_len,
        cross_attention_dim=256 // 4,
        attention_head_dim=8,       # number of heads or head dimension
        hidden_dim=128 // 4,
        down_block_types=(
            "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"
        ),
        up_block_types = (
            "UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"
        ),
        # ('UpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D') example

        # (np.array([128, 256, 256]) // 4).tolist()
        block_out_channels=(320 // div, 640 // div, 1280 // div, 1280 // div), # must match number of blocks above
    )


    if data_set.dataset_name == "MHD_64":
        model.input_format = lambda batch: batch.reshape(batch.shape[0], -1, * batch.shape[3:])
        model.cond_format = lambda cond: cond.reshape(cond.shape[0], -1, * cond.shape[3:])  # (B, Num Frames, F, Lx, Ly, Lz) -> (B, NF*F, Lx, Ly, Lz), works with 1 frame also

        def energyLoss_format(X0, cond, X_hat):

            len_fields = X0.shape[1]
            X_cat = torch.cat((X0, cond), dim=1)
            batch, channels, lx, ly, lz = X_cat.shape

            num_frames = channels // len_fields

            # Reshape to [batch, num_frames, len_fields, height, width]
            X_per_field = X_cat.view(batch, num_frames, len_fields, lx, ly, lz)
            X_mean = X_per_field.mean(dim=1)  # Average across frames

            u_p_p = (cond[:, :len_fields, :, :] - X_mean)[:, -3:, ...]
            u_p_t = (X0 - X_mean)[:, -3:, ...]
            u_p_h = (X_hat - X_mean)[:, -3:, ...]

            return u_p_p, u_p_t, u_p_h

        model.energyLoss_format = energyLoss_format


    return model

