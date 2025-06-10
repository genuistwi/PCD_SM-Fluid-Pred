# --- Custom_CondAttUnet model setup ---
from models.Custom_CondAttUnet.setup import model_setup as CC_att_setup

# --- Custom_CondAttUnet model setup + low noise skip gates ---
from models.Custom_CondAttUnet_lowSkip.setup import model_setup as CC_ls_att_setup
from models.Custom_CondAttUnet_lowSkip_3D.setup import model_setup as CC_ls_att_3D_setup

# --- Diffusers_UNet2DConditionModel models setup ---
from models.Diffusers_UNet2DConditionModel.setup import model_setup as DU_att_setup


def get_model(Cfg, data_set):

    modelCfg = Cfg.modelCfg
    model = None

    dataset_names = ["turbulent_radiative_layer_2D", "MHD_64", "JHTDB"]
    model_names = ["Att_Unet", "Custom_CondAttUnet_lowSkip", "Custom_CondAttUnet_lowSkip_3D", "UNet2DConditionModel"]
    assert data_set.dataset_name in dataset_names
    assert modelCfg.model_name in model_names
    assert modelCfg.conditional in [True, False]

    if modelCfg.model_name == "Att_Unet":
        model = CC_att_setup(Cfg, data_set)

    if modelCfg.model_name == "Custom_CondAttUnet_lowSkip":
        model = CC_ls_att_setup(Cfg, data_set)

    if modelCfg.model_name == "Custom_CondAttUnet_lowSkip_3D":
        model = CC_ls_att_3D_setup(Cfg, data_set)

    if modelCfg.model_name == "UNet2DConditionModel":
        model = DU_att_setup(Cfg, data_set)

    return model

