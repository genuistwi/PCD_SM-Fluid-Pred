from models.Custom_CondAttUnet.model import Network
import torch

def model_setup(Cfg, data_set):
    """
    Function called to create the model depending on the configuration but also the data set.
    Also sets two methods, input_format and output_format. It transforms the given batch and condition of the
    __getitem__  of the data set into the appropriate format for that specific model created.

    Supported: "Custom conditional Unet" and:
        - turbulent_radiative_layer_2D

    NB: the way it (input data channels) works is for data shaped [batch, field, dim_x, dim_y], physical time is
    therefore Omitted. For the conditioning, it is taken as multiple channels concatenated along fields, basically,
    [batch, field * number of frames, dim_x, dim_y].
    """

    assert data_set.dataset_name in ["turbulent_radiative_layer_2D", "JHTDB"]

    configKwargs = {
        "dim": 64,  # Dimension of positional time embedding, must be 2^n (hyperparameter)
        "init_dim": 10,  # Dimension encoder_img and encoder_cond (hyperparam)
        "dim_mults": (1, 2, 4, 4),
        "embedding": 'sinusoidal',  # Time embedding format
        "img_cond": 1 if Cfg.modelCfg.conditional else 0,
        "channels": data_set.fields_len,
        "cond_channels": int(data_set.fields_len * data_set.input_len) if Cfg.modelCfg.conditional else 0,
    }
    model = Network(**configKwargs)

    # Specific formatting for training, depends on the dataset.
    if data_set.dataset_name == "turbulent_radiative_layer_2D":
        """ 
        Works provided that "T Lx Ly F -> T F Lx Ly" reshaping got executed. When calling train/test/valLoader,
        we reshape following: [batch, num_frames, F, ...] -> [batch, num_frames * F, ...] to follow Unet's format.
        -> input: [batch, fields, dim_x, dim_y]
        -> cond: [batch, num_frames * fields, dim_x, dim_y]. 
        """
        model.input_format = lambda batch: batch.squeeze()  # We remove the dim = to "1" (1 frame)
        model.cond_format = lambda cond: cond.reshape(cond.shape[0], -1, * cond.shape[3:])

        """ 
        Used for energy loss regularization. We have:
        X0 = [batch, fields, dim_x, dim_y] -> frame f_n
        cond = [batch, num_frames * fields, dim_x, dim_y] -> increasing stack: f_n-i, f_n-i+1, ..., f_n-1
        """
        def energyLoss_format(X0, cond, X_hat):
            len_fields = X0.shape[1]
            X_cat = torch.cat((X0, cond), dim=1)
            batch, channels, height, width = X_cat.shape

            num_frames = channels // len_fields

            # Reshape to [batch, num_frames, len_fields, height, width]
            X_per_field = X_cat.view(batch, num_frames, len_fields, height, width)
            X_mean = X_per_field.mean(dim=1)  # Average across frames

            # Furthest frame is the first one ":len_fields"
            # u_p_p = "prime + past" // u_p_t = "prime + time tau" // u_p_h = "prime + time tau hat (estimate)"
            # u = U + u_p => u_p = u - U // last 2 fields for v_x and v_y (turbulent_radiative_layer_2D)

            u_p_p = (cond[:, :len_fields, :, :] - X_mean)[:, -2:, ...]
            u_p_t = (X0 - X_mean)[:, -2:, ...]
            u_p_h = (X_hat - X_mean)[:, -2:, ...]

            return u_p_p, u_p_t, u_p_h

        model.energyLoss_format = energyLoss_format

    if data_set.dataset_name == "JHTDB":
        model.input_format = lambda data: data.squeeze()
        model.cond_format = lambda cond: cond.reshape(cond.shape[0], -1, * cond.shape[3:])  # (B, NF, F, H, W) -> (B, NF*F, H, W), works with 1 frame also

    # else:
    #     model.input_format = lambda batch: batch
    #     model.cond_format = lambda cond: cond

    return model















