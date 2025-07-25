# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import torch
import numpy as np
import abc

from SDE.utils import get_score_fn
from scipy import integrate
from SDE import sde_lib
from models import utils as mutils


def match_dim(input, target):
    while len(input.shape) < len(target.shape):
        input = input.unsqueeze(-1)
    return input

def to_flattened_numpy(x):
    """ Flatten a torch tensor `x` and convert it to numpy """
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """ Form a torch tensor with the given `shape` from a flattened numpy array `x` """
    return torch.from_numpy(x.reshape(shape))



""" Various sampling methods, taken and adapted from https://arxiv.org/abs/2011.13456 """


_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """ A decorator for registering predictor classes """
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls
    return _register if cls is None else _register(cls)


def register_corrector(cls=None, *, name=None):
    """ A decorator for registering corrector classes """
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls
    return _register if cls is None else _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, eps):
    """Create a sampling function.

    Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
    """

    sampler_name = config.method
    # Probability flow ODE sampling with black-box ODE solvers

    samplerKwargs = dict(sde=sde, shape=shape, denoise=config.noise_removal, eps=eps)

    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(**samplerKwargs)

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.predictor.lower())
        corrector = get_corrector(config.corrector.lower())

        pcKwargs = dict(predictor=predictor, corrector=corrector, snr=config.snr, n_steps=config.n_steps_each,
                        probability_flow=config.probability_flow)
        sampling_fn = get_pc_sampler(**samplerKwargs, **pcKwargs)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, cond):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.
          cond: condition

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, cond):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.
          cond: condition

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, cond):

        dt = -self.rsde.T / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, cond)
        x_mean = x + drift * dt
        x = x_mean + match_dim(diffusion, x_mean) * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, cond):
        f, G = self.rsde.discretize(x, t, cond)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + match_dim(G, x_mean) * z
        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t, cond):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        sigma_diff = (sigma ** 2 - adjacent_sigma ** 2)

        score = self.score_fn(x, t, cond)
        x_mean = x + score * match_dim(sigma_diff, x)
        std = torch.sqrt((adjacent_sigma ** 2 * sigma_diff) / (sigma ** 2))
        noise = torch.randn_like(x)

        x = x_mean + match_dim(std, x) * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t, cond):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t, cond)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t, cond):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t, cond)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t, cond)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""
    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, cond):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, cond):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):  # flag
            grad = score_fn(x, t, cond)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """
    The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
    We include this corrector only for completeness.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, cond):  # flag
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)
        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, cond)
            noise = torch.randn_like(x)

            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + match_dim(step_size, x) * grad

            tt = torch.sqrt(step_size * 2)
            x = x_mean + noise * match_dim(tt, x_mean)

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """ An empty corrector that does nothing. """

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, cond):
        return x, x


def shared_predictor_update_fn(x, t, cond, sde, model, predictor, probability_flow):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = get_score_fn(model.Cfg, sde, model)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, cond)


def shared_corrector_update_fn(x, t, cond, sde, model, corrector, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = get_score_fn(model.Cfg, sde, model)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, cond)


def get_pc_sampler(sde, shape, predictor, corrector, snr,
                   n_steps=1, probability_flow=False, denoise=True, eps=1e-3):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

    Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictorKwargs = dict(sde=sde, predictor=predictor, probability_flow=probability_flow)
    predictor_update_fn = functools.partial(shared_predictor_update_fn, **predictorKwargs)

    correctorKwargs = dict(sde=sde, corrector=corrector, snr=snr, n_steps=n_steps)
    corrector_update_fn = functools.partial(shared_corrector_update_fn, **correctorKwargs)

    def pc_sampler(model, cond):

        device = cond.device

        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, cond=cond, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, cond=cond, model=model)

            return x_mean if denoise else x, sde.N * (n_steps + 1)  # no inverse scalar

    return pc_sampler


def get_ode_sampler(sde, shape, denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

    Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, cond):
        score_fn = get_score_fn(model.Cfg, sde, model)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, cond)
        return x

    def drift_fn(model, x, t, cond):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(model.Cfg, sde, model)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, cond)[0]

    def ode_sampler(model, z=None, cond=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """

        device = next(model.parameters()).device

        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, cond)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, cond)

            x = x  # no inverse scalar
            return x, nfe

    return ode_sampler


import torch.nn.functional as F
# ----------------------------------------------------------------------------------------------------------------------

def compute_divergence(fx, fy, dx, dy):
    # fx, fy: (Lx, Ly), implying Neumann BC
    div_x = (F.pad(fx[2:, :], (0, 0, 1, 1), mode='replicate') -
             F.pad(fx[:-2, :], (0, 0, 1, 1), mode='replicate')) / (
                2 * dx)
    div_y = (F.pad(fy[:, 2:], (1, 1, 0, 0), mode='replicate') -
             F.pad(fy[:, :-2], (1, 1, 0, 0), mode='replicate')) / (
                2 * dy)
    return div_x + div_y


def contEq_GD(rho_t, u_t, rho_pred, u_pred, dt, dx, dy, lr=1e-4, steps=10):
    """
    Improve rho_pred and u_pred at t+dt using mass conservation.

    rho_t:      (Lx, Ly) at time t
    u_t:        (2, Lx, Ly) at time t  [not directly used but could be]
    rho_pred:   (Lx, Ly) at t+dt (requires_grad=True)
    u_pred:     (2, Lx, Ly) at t+dt (requires_grad=True)
    """
    optimizer = torch.optim.SGD([rho_pred, u_pred], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        rho_u_x = rho_pred * u_pred[0]  # (Lx, Ly)
        rho_u_y = rho_pred * u_pred[1]  # (Lx, Ly)

        div_rho_u = compute_divergence(rho_u_x, rho_u_y, dx, dy)

        residual = (rho_pred - rho_t) / dt + div_rho_u

        loss = torch.mean(residual ** 2)

        loss.backward()
        optimizer.step()

    return rho_pred.detach(), u_pred.detach()
