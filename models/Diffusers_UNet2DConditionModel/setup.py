from models.Diffusers_UNet2DConditionModel.model import C_UNet2DConditionModel


def model_setup(Cfg, data_set):
    """
    Function called to create the model depending on the configuration but also the data set.
    Also sets two methods, input_format and output_format. It transforms the given batch and condition of the
    __getitem__  of the data set into the appropriate format for that specific model created.

    Supported: "Diffusers 2D conditional Unet" and:
        - turbulent_radiative_layer_2D

    NB: the way it (input data channels) works is for data shaped [batch, field, dim_x, dim_y], physical time is
    therefore Omitted. For the conditioning, it is taken as multiple channels concatenated along fields, basically,
    [batch, field * number of frames, dim_x, dim_y].
    """

    assert data_set.dataset_name in ["turbulent_radiative_layer_2D", "JHTDB"]

    configKwargs = {
        "in_channels": data_set.fields_len,  # Input image channels
        "condition_channels": int(data_set.fields_len * data_set.input_len) if Cfg.modelCfg.conditional else 0,
        "model_channels": 64,  # Number of base model channels
    }
    model = C_UNet2DConditionModel(**configKwargs)

    if data_set.dataset_name == "turbulent_radiative_layer_2D":
        model.input_format = lambda data: data.squeeze()
        model.cond_format = lambda cond: cond.reshape(cond.shape[0], -1, * cond.shape[3:])  # (B, NF, F, H, W) -> (B, NF*F, H, W), works with 1 frame also

    if data_set.dataset_name == "JHTDB":
        model.input_format = lambda data: data.squeeze()
        model.cond_format = lambda cond: cond.reshape(cond.shape[0], -1, * cond.shape[3:])  # (B, NF, F, H, W) -> (B, NF*F, H, W), works with 1 frame also


    return model
