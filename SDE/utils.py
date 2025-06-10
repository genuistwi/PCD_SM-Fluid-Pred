from SDE import sde_lib
from utils.general import match_dim
import torch


def get_sde(Cfg):
    sdeCfg = Cfg.sdeCfg
    sde = None
    assert sdeCfg.sde_name in ["vpsde", "subvpsde", "vesde", ]

    timeKwargs = {
        "N": sdeCfg.num_scales,
        "T": sdeCfg.T_max,
    }

    if sdeCfg.sde_name == "vpsde":
        sde = sde_lib.VPSDE(beta_min=sdeCfg.beta_min, beta_max=sdeCfg.beta_max, **timeKwargs)
    elif sdeCfg.sde_name == "subvpsde":
        sde = sde_lib.subVPSDE(beta_min=sdeCfg.beta_min, beta_max=sdeCfg.beta_max, **timeKwargs)
    elif sdeCfg.sde_name == "vesde":
        sde = sde_lib.VESDE(sigma_min=sdeCfg.sigma_min, sigma_max=sdeCfg.sigma_max, **timeKwargs)

    return sde


def get_score_fn(Cfg, SDE, model):

    assert Cfg.modelCfg.model_name in ["Att_Unet", "Custom_CondAttUnet_lowSkip", "Custom_CondAttUnet_lowSkip_3D", "UNet2DConditionModel"]

    def score_fn(Xt, t, cond):

        # # Time embedding in typical Unet denoiser range up 1000 timesteps
        # t_normalized = t/SDE.T  # get back the [0, 1] range
        # t_embedding = torch.round(t_normalized * (SDE.N - 1)).long()
        # # FLAG to do: adjust sampling procedure

        if Cfg.modelCfg.model_name == "Att_Unet":
            t_normalized = t / SDE.T  # get back the [0, 1] range
            t_embedding = torch.round(t_normalized * (SDE.N - 1)).long()

            model_output = model(Xt, t_embedding, cond)

            """ Transforms a denoising model into a score function. """
            # var_t = torch.square(SDE.marginal_prob(torch.zeros_like(Xt), t)[1])
            std_t = SDE.marginal_prob(torch.zeros_like(Xt), t)[1]

            std_t[std_t.abs() < 1e-5] = 1e-4
            std_t = match_dim(std_t, model_output)

            score = - model_output / std_t

            return score

        if Cfg.modelCfg.model_name == "Custom_CondAttUnet_lowSkip":
            t_normalized = t / SDE.T  # get back the [0, 1] range
            std_t = SDE.marginal_prob(torch.zeros_like(Xt), t)[1]

            # if Cfg.sdeCfg.sde_name in ["vpsde", "subvpsde"]:

            t_embedding = torch.round(t_normalized * (SDE.N - 1)).long()

            model_output = model(Xt, t_embedding, cond, std_t)
            std_t[std_t.abs() < 1e-5] = 1e-4
            score = - model_output / match_dim(std_t, model_output)

            return score

        if Cfg.modelCfg.model_name == "Custom_CondAttUnet_lowSkip_3D":
            t_normalized = t / SDE.T  # get back the [0, 1] range
            std_t = SDE.marginal_prob(torch.zeros_like(Xt), t)[1]

            # if Cfg.sdeCfg.sde_name in ["vpsde", "subvpsde"]:

            t_embedding = torch.round(t_normalized * (SDE.N - 1)).long()

            model_output = model(Xt, t_embedding, cond, std_t)
            std_t[std_t.abs() < 1e-5] = 1e-4
            score = - model_output / match_dim(std_t, model_output)

            # elif Cfg.sdeCfg.sde_name in ["vesde"]:
            #     t_embedding = SDE.marginal_prob(torch.zeros_like(Xt), t)[1]
            #
            #     model_output = model(Xt, t_embedding, cond, std_t)
            #     score = model_output / std_t

            # else:
            #     raise NotImplementedError("SDE not recognized.")

            return score

        if Cfg.modelCfg.model_name == "UNet2DConditionModel":

            t_normalized = t / SDE.T  # get back the [0, 1] range
            t_embedding = torch.round(t_normalized * (SDE.N - 1)).long()

            model_output = model(Xt, t_embedding, cond)

            """ Transforms a denoising model into a score function. """
            # var_t = torch.square(SDE.marginal_prob(torch.zeros_like(Xt), t)[1])
            std_t = SDE.marginal_prob(torch.zeros_like(Xt), t)[1]
            score = - model_output / match_dim(std_t, model_output)

            return score

    return score_fn
