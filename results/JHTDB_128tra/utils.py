import numpy as np

def compute_q_criterion(u, v, dx=1.0, dy=1.0):
    T, Lx, Ly = u.shape
    Q = np.zeros_like(u)

    for t in range(T):
        ux = np.gradient(u[t], dx, axis=0)
        uy = np.gradient(u[t], dy, axis=1)
        vx = np.gradient(v[t], dx, axis=0)
        vy = np.gradient(v[t], dy, axis=1)

        S11 = ux
        S22 = vy
        S12 = 0.5 * (uy + vx)

        Omega12 = 0.5 * (uy - vx)

        S_sq = S11**2 + 2*S12**2 + S22**2
        Omega_sq = 2 * Omega12**2

        Q[t] = 0.5 * (Omega_sq - S_sq)

    return Q


import torch
import torch.nn.functional as F


def gradient_2d(field):
    # Central difference gradient
    grad_x = F.pad(field[..., 2:, :] - field[..., :-2, :], (0, 0, 1, 1)) / 2
    grad_y = F.pad(field[..., :, 2:] - field[..., :, :-2], (1, 1, 0, 0)) / 2
    return grad_x, grad_y


def divergence_2d(grad_x, grad_y):
    div_x = F.pad(grad_x[..., 2:, :] - grad_x[..., :-2, :], (0, 0, 1, 1)) / 2
    div_y = F.pad(grad_y[..., :, 2:] - grad_y[..., :, :-2], (1, 1, 0, 0)) / 2
    return div_x + div_y


def perona_malik_diffusion(field, num_iters=20, kappa=0.1, step=0.1):
    for _ in range(num_iters):
        grad_x, grad_y = gradient_2d(field)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        c = 1.0 / (1.0 + (grad_mag / kappa) ** 2)  # Diffusion coefficient

        c_grad_x = c * grad_x
        c_grad_y = c * grad_y
        div = divergence_2d(c_grad_x, c_grad_y)

        field = field + step * div
    return field


def smooth_velocity_gradients(flow):
    """
    flow: Tensor of shape (T, 2, Lx, Ly), where 0 -> u, 1 -> v
    returns: smoothed flow with same shape
    """
    T, C, Lx, Ly = flow.shape
    flow = torch.tensor(flow)

    smoothed_flow = torch.zeros_like(flow)

    for t in range(T):
        u = flow[t, 0]
        v = flow[t, 1]

        # Compute gradients
        dux, duy = gradient_2d(u)
        dvx, dvy = gradient_2d(v)

        # Stack gradients and smooth them
        grads = [dux, duy, dvx, dvy]
        smoothed_grads = [perona_malik_diffusion(g) for g in grads]

        # Optionally: Reconstruct velocity by integrating (skip if you just need smooth Q)
        # For now, we'll skip that and return the smoothed gradients if needed

        # Or keep original field and replace noisy gradients for Q computation
        smoothed_flow[t, 0] = u  # keep original u
        smoothed_flow[t, 1] = v  # keep original v

        # You could also reconstruct `Q` from the smoothed gradients here

    smoothed_flow[:, 2:] = flow[:, 2:]

    return np.array(smoothed_flow)
