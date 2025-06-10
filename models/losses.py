import torch
import numpy as np
from utils.general import *

import matplotlib.pyplot as plt
import random


def get_loss_fn(SDE, score_fn, eps=1e-5):


    def loss_fn(X0, cond):

        t = torch.rand((X0.shape[0], 1), device=X0.device) * (SDE.T - eps) + eps
        mu_x, std = SDE.marginal_prob(X0, t)
        std = match_dim(std, X0)

        Zt = torch.randn_like(X0)
        Xt = mu_x + std * Zt

        score = score_fn(Xt, t, cond)

        # --- basic ---
        losses = torch.square(score * std + Zt)
        losses = losses.reshape(losses.shape[0], -1)

        # --- basic ---
        loss_avg = torch.mean(losses, dim=-1)  # average across all pixels
        loss_tot = torch.mean(loss_avg)  # average across batches

        gauss_mean, gauss_std = SDE.marginal_coeffs(X0, t)
        gauss_mean = match_dim(gauss_mean, X0)
        X_hat = ((Xt + std**2 * score) / gauss_mean)

        return loss_tot, X_hat

    return loss_fn


def get_energyLoss_fn(dim):

    def energyLoss_fn(u_p_p, u_p_t, u_p_h):

        # E_r1,r2 { || u_tau-tn || x || u_tau - û_tau || }

        if dim == 2:
            batch_size, channels, Lx, Ly = u_p_p.shape
        elif dim == 3:
            batch_size, channels, Lx, Ly, Lz = u_p_p.shape

        # 1) Compute the point-wise squared L2 norm of u_p_p(r):
        u_sq = (u_p_p ** 2).sum(dim=1)  # [batch, Lx, Ly, ...]

        # 2) Compute the point-wise squared L2 norm of [u_a(r) - u_b(r)]:
        diff_ab_sq = ((u_p_t - u_p_h) ** 2).sum(dim=1)  # [batch, Lx, Ly, ...]

        # 3) Sum over all spatial locations for each item in the batch
        sum_u_sq = u_sq.sum(dim=[1,2] if dim == 2 else [1,2,3])  # [batch]
        sum_diff_ab_sq = diff_ab_sq.sum(dim=[1,2] if dim == 2 else [1,2,3])  # [batch]

        # 4) Multiply them => total double-sum of the outer products
        outer_sum = sum_u_sq * sum_diff_ab_sq  # [batch]

        # 5) average over all {r1,r2} pairs:
        n_r = Lx * Ly  if dim == 2 else Lx * Ly * Lz # total # of r points
        outer_mean = outer_sum / float(n_r ** 2)  # [batch]

        # 6) Then average over the batch dimension => final scalar
        expectation = outer_mean.mean()

        return expectation

    return energyLoss_fn


def get_val_loss_fn(SDE, score_fn, eps=1e-5):

    def val_loss_fn(X0, cond):

        t = torch.rand((X0.shape[0], 1), device=X0.device) * (SDE.T - eps) + eps
        mu_x, std = SDE.marginal_prob(X0, t)
        std = match_dim(std, X0)

        gauss_mean, _ = SDE.marginal_coeffs(X0, t)
        gauss_mean = match_dim(gauss_mean, X0)

        Zt = torch.randn_like(X0)
        Xt = mu_x + std * Zt

        score = score_fn(Xt, t, cond)

        X_hat = ((Xt + std**2 * score) / gauss_mean)

        MSE_ref = torch.mean((X0 - Xt) ** 2)
        MSE_est = torch.mean((X0 - X_hat) ** 2)

        losses = torch.square(score * std + Zt)
        losses = losses.reshape(losses.shape[0], -1)
        loss_avg = torch.mean(losses, dim=-1)  # average across all pixels
        loss_tot = torch.mean(loss_avg)  # average across batches

        return loss_tot, MSE_ref, MSE_est

    return val_loss_fn
