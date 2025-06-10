""" Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Taken and adapted from https://arxiv.org/abs/2011.13456. """

import abc
from abc import ABC

import torch
import numpy as np


def match_dim(input, target):
    while len(input.shape) < len(target.shape):
        input = input.unsqueeze(-1)
    return input


class SDE(abc.ABC):
    def __init__(self, N, T):
        super().__init__()
        self.N = N
        self.T = T
        self.dt = self.T / self.N

    @abc.abstractmethod
    def sde(self, x, t, cond):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$. """
        pass

    @abc.abstractmethod
    def marginal_coeffs(self, x, t):
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
        Sends back only the mean and var. """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """ Generate one sample from the prior distribution, $p_T(x)$. """
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """ Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density """
        pass

    def discretize(self, x, t, cond=None):
        """ Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization. """

        drift, diffusion = self.sde(x, t, cond)
        f = drift * self.dt
        G = diffusion * torch.sqrt(torch.tensor(self.dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):

        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize
        marginal_coeffs = self.marginal_coeffs

        class RSDE(self.__class__, ABC):
            """ Build the class for reverse-time SDE. """

            def __init__(self):

                self.N = N
                self.T = T

                self.score_fn = score_fn
                self.discretize_fn = discretize_fn
                self.probability_flow = probability_flow
                self.marginal_coeffs = marginal_coeffs

            def sde(self, x, t, cond=None):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t, cond=cond)
                score = self.score_fn(x, t, cond)
                drift = drift - match_dim(diffusion, drift) ** 2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, cond=None):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = self.discretize_fn(x, t)

                rev_f = f - match_dim(G, f) ** 2 * self.score_fn(x, t, cond) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, T=1):

        super().__init__(N, T)

        self.N = N
        self.T = T

        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)


    def sde(self, x, t, cond=None):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * match_dim(beta_t, x) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(match_dim(log_mean_coeff, x)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def marginal_coeffs(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t, cond=None):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = match_dim(torch.sqrt(alpha), x) * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, T=1):

        super().__init__(N, T)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = T

        # # --- perso --- ???
        # self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        # self.alphas = 1. - self.discrete_betas

    def sde(self, x, t, cond=None):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * match_dim(beta_t, x) * x
        exp_diff = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * exp_diff)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = match_dim(torch.exp(log_mean_coeff), x) * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def marginal_coeffs(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, T=1):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.T = T

    def sde(self, x, t, cond=None):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        log_diff = np.log(self.sigma_max) - np.log(self.sigma_min)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * log_diff, device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def marginal_coeffs(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = torch.tensor([1.], device=x.device)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t, cond=None):
        """ SMLD(NCSN) discretization. """
        device = x.device
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(device)[timestep]
        discrete_sigmas = self.discrete_sigmas.to(device)[timestep - 1]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t, device=device), discrete_sigmas)
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
