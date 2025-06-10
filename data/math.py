import torch


def central_diff_roll(var: torch.Tensor, dim: int, spacing=1.0) -> torch.Tensor:
    """
    Finite difference derivative using torch.roll to shift forward/backward.
    Implies periodic boundaries along 'dim'.
    """
    var_p = var.roll(shifts=-1, dims=dim)  # shift forward
    var_m = var.roll(shifts=+1, dims=dim)  # shift backward

    deriv = (var_p - var_m) / (2.0 * spacing)  # central difference
    return deriv

def compute_velocity_gradients_3D(u, v, w, dx=1., dy=1., dz=1.):
    """
    Compute partial derivatives w.r.t x, y, z

    u, v, w, velocity x, y and z, shaped [time, Lx, Ly, Lz]
    dx, dy, dz: grid spacing in x,y,z.

    Returns a dict of partial derivatives, each shaped [time, Lx, Ly, Lz].
    """

    u_x = central_diff_roll(u, dim=-3, spacing=dx)
    u_y = central_diff_roll(u, dim=-2, spacing=dy)
    u_z = central_diff_roll(u, dim=-1, spacing=dz)

    v_x = central_diff_roll(v, dim=-3, spacing=dx)
    v_y = central_diff_roll(v, dim=-2, spacing=dy)
    v_z = central_diff_roll(v, dim=-1, spacing=dz)

    w_x = central_diff_roll(w, dim=-3, spacing=dx)
    w_y = central_diff_roll(w, dim=-2, spacing=dy)
    w_z = central_diff_roll(w, dim=-1, spacing=dz)

    return {
        "u_x": u_x, "u_y": u_y, "u_z": u_z,
        "v_x": v_x, "v_y": v_y, "v_z": v_z,
        "w_x": w_x, "w_y": w_y, "w_z": w_z,
    }

def compute_velocity_gradients_2D(u, v, dx=1., dy=1.):

    u_x, u_y = central_diff_roll(u, dim=-2, spacing=dx), central_diff_roll(u, dim=-1, spacing=dy)
    v_x, v_y = central_diff_roll(v, dim=-2, spacing=dx), central_diff_roll(v, dim=-1, spacing=dy)

    return {
        "u_x": u_x, "u_y": u_y,
        "v_x": v_x, "v_y": v_y,
    }

def compute_q_criterion(u, v, w=None, dx=1., dy=1., dz=1.):

    if w is None:
        grads = compute_velocity_gradients_2D(u, v, dx, dy)

        u_x, u_y = grads["u_x"], grads["u_y"]
        v_x, v_y = grads["v_x"], grads["v_y"]

        Sxx, Syy = u_x, v_y
        Sxy = 0.5 * (u_y + v_x)

        Om_xy = 0.5 * (u_y - v_x)
        S_dot_S = (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2)

        # Ω:Ω = 2 Om_xy^2
        Om_dot_Om = 2.0 * (Om_xy ** 2)

        # Q = 1/2 (Ω:Ω  - S:S)
        Q = 0.5 * (Om_dot_Om - S_dot_S)

    else:
        grads = compute_velocity_gradients_3D(u, v, w, dx, dy, dz)

        # Components of S (symmetric):
        # Diagonal
        Sxx = grads["u_x"]
        Syy = grads["v_y"]
        Szz = grads["w_z"]

        # Off‐diagonal: Sxy = 0.5(u_y + v_x), etc.
        Sxy = 0.5 * (grads["u_y"] + grads["v_x"])
        Sxz = 0.5 * (grads["u_z"] + grads["w_x"])
        Syz = 0.5 * (grads["v_z"] + grads["w_y"])

        # Components of Ω (antisymmetric):
        # Diagonal terms are zero for a strictly antisymmetric 3x3
        Omxy = 0.5 * (grads["u_y"] - grads["v_x"])
        Omxz = 0.5 * (grads["u_z"] - grads["w_x"])
        Omyz = 0.5 * (grads["v_z"] - grads["w_y"])

        # Now compute S:S = sum(Sij^2) and Ω:Ω = sum(Ωij^2).
        # For a 3×3, S:S = Sxx^2 + Syy^2 + Szz^2 + 2*(Sxy^2 + Sxz^2 + Syz^2).
        # Similarly for Ω:Ω.
        S_dot_S = (Sxx ** 2 + Syy ** 2 + Szz ** 2 + 2 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2))
        Om_dot_Om = (2 * (Omxy ** 2 + Omxz ** 2 + Omyz ** 2))  # factor 2: each off‐diag appears twice in a 3×3 sum

        # Q = 1/2 (Ω:Ω - S:S)
        Q = 0.5 * (Om_dot_Om - S_dot_S)

    return Q
