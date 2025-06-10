import numpy as np
from scipy.ndimage import convolve
from scipy.stats import wasserstein_distance

import copy

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def compute_mag_U_frequency_series(time_series):

    # Get the shape of the input time series
    time_dim = time_series.shape[0]

    # Perform the Fourier transform along the time axis
    U_f = np.fft.fft(time_series, axis=0)

    # Get the frequencies corresponding to the Fourier components
    frequencies = np.fft.fftfreq(time_dim)

    # Create a dictionary to store the results
    mag_U_dict = {}

    # Loop over the frequencies and compute the magnitude of U for each
    for idx, f in enumerate(frequencies):
        # Compute the magnitude (absolute value) of the Fourier-transformed U at frequency f
        mag_U_f = np.abs(U_f[idx])

        # Dynamically determine which axes are spatial (all axes except the time axis)
        spatial_axes = tuple(range(1, len(time_series.shape)))  # This includes all axes except axis 0 (time axis)

        # Spatially average the magnitude across all spatial dimensions (excluding the time axis)
        mag_U_spatial_avg = np.mean(mag_U_f, axis=spatial_axes)

        # Store the result in the dictionary
        mag_U_dict[f] = mag_U_spatial_avg

    return mag_U_dict



def pearson_correlation_over_time(simu, prediction):
    time_steps = simu.shape[0]
    correlations = np.zeros(time_steps)

    for t in range(time_steps):
        x = simu[t].ravel()
        y = prediction[t].ravel()
        if np.std(x) == 0 or np.std(y) == 0:
            correlations[t] = np.nan  # Avoid division by zero if constant
        else:
            correlations[t] = pearsonr(x, y)[0]

    return correlations


def spearman_correlation_over_time(simu, prediction):
    time_steps = simu.shape[0]
    correlations = np.zeros(time_steps)

    for t in range(time_steps):
        x = simu[t].ravel()
        y = prediction[t].ravel()
        if np.std(x) == 0 or np.std(y) == 0:
            correlations[t] = np.nan  # Undefined if constant
        else:
            correlations[t] = spearmanr(x, y).correlation

    return correlations


def compute_relative_error(GT, pred, epsilon=1e-9):
    spatial_axes = tuple(range(1, pred.ndim))  # All axes except time
    num = np.sqrt(np.sum((GT - pred) ** 2, axis=spatial_axes))
    denom = np.sqrt(np.sum(GT ** 2, axis=spatial_axes))
    return num / (denom + epsilon)  # Avoid division by zero



# ---------- KL div----------
def kl_divergence(p, q):
    # Convert to float and normalize histograms to get probability distributions
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Add a small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # Normalize
    p /= np.sum(p)
    q /= np.sum(q)

    return np.sum(p * np.log(p / q))


# ---------- 2D curl----------
def compute_2Dcurl(u, v, dx=1.0, dy=1.0):
    # Assume u, v shape: (time, Lx, Ly)
    dv_dx = np.gradient(v, dx, axis=1)  # ∂v/∂x
    du_dy = np.gradient(u, dy, axis=2)  # ∂u/∂y
    curl = dv_dx - du_dy
    return curl  # shape: (time, Lx, Ly)

def compute_3Dcurl(u, v, w, dx=1.0, dy=1.0, dz=1.0):
    # Assume shape: (T, Lx, Ly, Lz)
    dw_dy = np.gradient(w, dy, axis=2)
    dv_dz = np.gradient(v, dz, axis=3)
    du_dz = np.gradient(u, dz, axis=3)
    dw_dx = np.gradient(w, dx, axis=1)
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=2)

    curl_x = dw_dy - dv_dz  # ∂w/∂y - ∂v/∂z
    curl_y = du_dz - dw_dx  # ∂u/∂z - ∂w/∂x
    curl_z = dv_dx - du_dy  # ∂v/∂x - ∂u/∂y

    return curl_x, curl_y, curl_z  # Each of shape (T, Lx, Ly, Lz)


# ---------- Q criterion ----------
def compute_2Dq_criterion(u, v, dx=1.0, dy=1.0):
    # Assume u, v shape: (T, Lx, Ly)
    ux = np.gradient(u, dx, axis=1)
    uy = np.gradient(u, dy, axis=2)
    vx = np.gradient(v, dx, axis=1)
    vy = np.gradient(v, dy, axis=2)

    S11 = ux
    S22 = vy
    S12 = 0.5 * (uy + vx)
    Omega12 = 0.5 * (uy - vx)

    S_sq = S11**2 + 2 * S12**2 + S22**2
    Omega_sq = 2 * Omega12**2

    Q = 0.5 * (Omega_sq - S_sq)
    return Q


def compute_3Dq_criterion(u, v, w, dx=1.0, dy=1.0, dz=1.0):
    # Assume shape: (T, Lx, Ly, Lz)
    ux = np.gradient(u, dx, axis=1)
    uy = np.gradient(u, dy, axis=2)
    uz = np.gradient(u, dz, axis=3)

    vx = np.gradient(v, dx, axis=1)
    vy = np.gradient(v, dy, axis=2)
    vz = np.gradient(v, dz, axis=3)

    wx = np.gradient(w, dx, axis=1)
    wy = np.gradient(w, dy, axis=2)
    wz = np.gradient(w, dz, axis=3)

    # Symmetric strain rate tensor S_ij
    S11 = ux
    S22 = vy
    S33 = wz
    S12 = 0.5 * (uy + vx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)

    # Antisymmetric vorticity tensor Omega_ij
    Omega12 = 0.5 * (uy - vx)
    Omega13 = 0.5 * (uz - wx)
    Omega23 = 0.5 * (vz - wy)

    S_sq = S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2)
    Omega_sq = 2 * (Omega12**2 + Omega13**2 + Omega23**2)

    Q = 0.5 * (Omega_sq - S_sq)
    return Q  # shape: (T, Lx, Ly, Lz)


# ---------- Smooth flow anisotropic ----------
dx_kernel = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
dy_kernel = dx_kernel.T

def gradient_2d(field):
    grad_x,grad_y = convolve(field, dx_kernel, mode='nearest'), convolve(field, dy_kernel, mode='nearest')
    return grad_x, grad_y
def divergence_2d(grad_x, grad_y):
    div_x,div_y = convolve(grad_x, dx_kernel, mode='nearest'), convolve(grad_y, dy_kernel, mode='nearest')
    return div_x + div_y
def Perona_malik(field, num_iters=20, kappa=0.1, step=0.1):
    for _ in range(num_iters):
        grad_x, grad_y = gradient_2d(field)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        c = 1.0 / (1.0 + (grad_mag / kappa)**2)
        c_grad_x, c_grad_y = c * grad_x, c * grad_y
        div = divergence_2d(c_grad_x, c_grad_y)
        field += step * div
    return field
def smooth_flow_anisotropic(u, v, num_iters=25, kappa=0.1, step=0.1):
    T, Lx, Ly = u.shape

    for t in range(T):
        u[t] = Perona_malik(u[t].copy(), num_iters, kappa, step)
        v[t] = Perona_malik(v[t].copy(), num_iters, kappa, step)
    return u, v


# ---------- TKE spectra ----------

def Energy_spectra(u, v, w=None, eps=1e-10):
    if w is None:
        dim = 2
        U = u
        V = v
        W = np.zeros_like(u)
    else:
        dim = 3
        U = u
        V = v
        W = w

    # Compute FFT amplitudes and average over time
    amplsU = np.abs(np.fft.fftn(U, axes=tuple(range(1, U.ndim))) / np.prod(U.shape[1:])).mean(axis=0)
    amplsV = np.abs(np.fft.fftn(V, axes=tuple(range(1, V.ndim))) / np.prod(V.shape[1:])).mean(axis=0)
    amplsW = np.abs(np.fft.fftn(W, axes=tuple(range(1, W.ndim))) / np.prod(W.shape[1:])).mean(axis=0)

    EK_U = amplsU**2
    EK_V = amplsV**2
    EK_W = amplsW**2

    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)
    EK_W = np.fft.fftshift(EK_W)

    shape = EK_U.shape
    box_radius = int(np.ceil(np.linalg.norm(shape) / 2)) + 1

    center = tuple(s // 2 for s in shape)

    EK_U_avsphr = np.zeros(box_radius) + eps
    EK_V_avsphr = np.zeros(box_radius) + eps
    EK_W_avsphr = np.zeros(box_radius) + eps

    it = np.ndindex(shape)
    for idx in it:
        wn = int(np.round(np.sqrt(sum((i - c)**2 for i, c in zip(idx, center)))))
        if wn < box_radius:
            EK_U_avsphr[wn] += EK_U[idx]
            EK_V_avsphr[wn] += EK_V[idx]
            EK_W_avsphr[wn] += EK_W[idx]

    EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)

    realsize = box_radius
    k = np.arange(realsize)
    E_k = EK_avsphr[:realsize]

    return {'k': k, 'E_k': E_k}


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
# ---------- Processing ----------

def GT_measures(dict_in, fields_name, n_bins, dataset_name=None, dim=2):

    assert dataset_name in ["JHTDB", "turbulent_radiative_layer_2D", "MHD_64"]
    dim = 3 if dataset_name == "MHD_64" else 2

    # --- velocity extraction ---
    if dataset_name == "JHTDB":
        u = dict_in["GT"]["pred"][:, 0]
        v = dict_in["GT"]["pred"][:, 1]
    elif dataset_name == "turbulent_radiative_layer_2D":
        u = dict_in["GT"]["pred"][:, 2]
        v = dict_in["GT"]["pred"][:, 3]
    elif dataset_name == "MHD_64":
        u = dict_in["GT"]["pred"][:, -3]
        v = dict_in["GT"]["pred"][:, -2]
        w = dict_in["GT"]["pred"][:, -1]

        Bx = dict_in["GT"]["pred"][:, 1]
        By = dict_in["GT"]["pred"][:, 2]
        Bz = dict_in["GT"]["pred"][:, 3]

    else:
        raise ValueError("Unknown dataset name")

    if dim == 2:
        dict_in["GT"]["u"] = copy.deepcopy(u)
        dict_in["GT"]["v"] = copy.deepcopy(v)
        dict_in["GT"]["spectra"] = Energy_spectra(u, v)
        dict_in["GT"]["mag_U"] = np.sqrt(np.square(u) + np.square(v))
    else:
        dict_in["GT"]["u"] = copy.deepcopy(u)
        dict_in["GT"]["v"] = copy.deepcopy(v)
        dict_in["GT"]["w"] = copy.deepcopy(w)

        dict_in["GT"]["mag_U"] = np.sqrt(np.square(u) + np.square(v) + np.square(w))
        dict_in["GT"]["spectra"] = Energy_spectra(u, v, w)

        if dataset_name == "MHD_64":
            dict_in["GT"]["Bx"] = copy.deepcopy(Bx)
            dict_in["GT"]["By"] = copy.deepcopy(By)
            dict_in["GT"]["Bz"] = copy.deepcopy(Bz)

            dict_in["GT"]["mag_spectra"] = Energy_spectra(Bx, By, Bz)
            dict_in["GT"]["mag_B"] = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))

    # --- stats ---
    dict_in["GT"]["time_mean"] = np.mean(dict_in["GT"]["pred"], axis=0)
    dict_in["GT"]["time_std"] = np.std(dict_in["GT"]["pred"], axis=0)

    if dim == 2:
        dict_in["GT"]["spatial_mean"] = np.mean(dict_in["GT"]["pred"], axis=(2,3))  # Collapse on lx,ly
        dict_in["GT"]["spatial_std"] = np.std(dict_in["GT"]["pred"], axis=(2,3))
        dict_in["GT"]["spatial_std_mag_U"] = np.std(dict_in["GT"]["mag_U"], axis=(1,2))

        # --- phi properties ---
        dict_in["GT"]["curl"] = compute_2Dcurl(u, v, dx=1., dy=1.)
        dict_in["GT"]["q_crit"] = compute_2Dq_criterion(u, v, dx=1., dy=1.)
    else:
        dict_in["GT"]["spatial_mean"] = np.mean(dict_in["GT"]["pred"], axis=(2, 3, 4))  # Collapse on lx,ly,lz
        dict_in["GT"]["spatial_std"] = np.std(dict_in["GT"]["pred"], axis=(2, 3, 4))
        dict_in["GT"]["spatial_std_mag_U"] = np.std(dict_in["GT"]["mag_U"], axis=(1,2,3))


        # --- phi properties ---
        dict_in["GT"]["curl"] = compute_3Dcurl(u, v, w, dx=1., dy=1., dz=1.)
        dict_in["GT"]["q_crit"] = compute_3Dq_criterion(u, v, w, dx=1., dy=1., dz=1.)

    # dict_in["GT"]["mag_U_fft"] = compute_mag_U_frequency_series(dict_in["GT"]["mag_U"])


    # --- per field histogram ---
    dict_in["GT"]["time_histogram"], dict_in["GT"]["spatial_histogram"], dict_in["GT"]["support"] = {}, {}, {}
    for field in range(len(fields_name)):
        pred_field = dict_in["GT"]["pred"][:,field]

        if dim == 2:
            T, Lx, Ly = pred_field.shape
        else:
            T, Lx, Ly, Lz = pred_field.shape

        # global_min = pred_field.min()
        # global_max = pred_field.max()
        mean = np.mean(pred_field)
        global_min = mean - np.abs(mean)/3
        global_max = mean + np.abs(mean)/3
        bin_edges_space = np.linspace(global_min, global_max, n_bins+1)

        histKwargs = dict(bins=bin_edges_space, density=True)
        # if dim == 2:
        #     dict_in["GT"]["time_histogram"][field] = np.zeros((Lx, Ly, n_bins), dtype=np.int64)
        # else:
        #     dict_in["GT"]["time_histogram"][field] = np.zeros((Lx, Ly, Lz, n_bins), dtype=np.int64)

        dict_in["GT"]["spatial_histogram"][field] = np.zeros((T, n_bins), dtype=np.int64)

        # for ix in range(Lx):  # Time histo, (Lx, Ly, support) -> plot(
        #     for iy in range(Ly):
        #         time_series = pred_field[:, ix, iy]
        #         hist_counts, _ = np.histogram(time_series, **histKwargs)
        #         dict_in["GT"]["time_histogram"][field][ix, iy, :] = hist_counts
        for t in range(T):  # Spatial histo, (t, support) -> plot(support, t=x) observe field's average value at a given time
            spatial_values = pred_field[t].ravel()
            hist_counts, _ = np.histogram(spatial_values, **histKwargs)
            dict_in["GT"]["spatial_histogram"][field][t, :] = hist_counts

        # _, dict_in["GT"]["support"][field] = np.histogram(pred_field, bins=bins, density=True, range=(np.min(pred_field), np.max(pred_field)))
        dict_in["GT"]["support"][field] = bin_edges_space


    q_crit = dict_in["GT"]["q_crit"]
    abs_curl = np.abs(dict_in["GT"]["curl"])

    if dim == 2:
        T, Lx, Ly = q_crit.shape
    else:
        T, Lx, Ly, Lz = q_crit.shape

    q_crit_positive_vals = q_crit[q_crit > 0.]
    p95 = np.percentile(q_crit_positive_vals, 99)

    bins_q_crit = np.linspace(1e-8, p95, n_bins + 1)
    bins_curl = np.linspace(0, np.max(abs_curl), n_bins + 1)

    dict_in["GT"]["curl_spatial_histogram"] = np.zeros((T, n_bins), dtype=np.int64)
    dict_in["GT"]["q_crit_spatial_histogram"] = np.zeros((T, n_bins), dtype=np.int64)

    for t in range(T):
        hist_q_crit, _ = np.histogram(q_crit[t].ravel(), bins=bins_q_crit, density=True)
        dict_in["GT"]["q_crit_spatial_histogram"][t, :] = hist_q_crit
        if dim == 2:
            hist_curl, _ = np.histogram(abs_curl[t].ravel(), bins=bins_curl, density=True)
            dict_in["GT"]["curl_spatial_histogram"][t, :] = hist_curl  # In 3D, it becomes a vector...


    dict_in["GT"]["q_crit_support"] = bins_q_crit
    dict_in["GT"]["curl_support"] = bins_curl


def pred_measures(dict_in, version, regularization, sde_names, fields_name, n_bins,
                  perona=False, dataset_name=None, dim=2):

    assert dataset_name in ["JHTDB", "turbulent_radiative_layer_2D", "MHD_64"]
    dim = 3 if dataset_name == "MHD_64" else 2

    for regu in regularization:
        for sde in sde_names:

            try:
                dict_in[regu][sde]["u"][version] = None
                dict_in[regu][sde]["v"][version] = None
                dict_in[regu][sde]["w"][version] = None
                dict_in[regu][sde]["Bx"][version] = None
                dict_in[regu][sde]["By"][version] = None
                dict_in[regu][sde]["Bz"][version] = None
                dict_in[regu][sde]["mag_U"][version] = None
                dict_in[regu][sde]["mag_B"][version] = None

                dict_in[regu][sde]["RE"][version] = None
                dict_in[regu][sde]["RE_U"][version] = None
                dict_in[regu][sde]["RE_B"][version] = None
                dict_in[regu][sde]["RE_mag_U"][version] = None
                dict_in[regu][sde]["RE_mag_B"][version] = None

                dict_in[regu][sde]["spectra"][version] = None
                dict_in[regu][sde]["mag_spectra"][version] = None

                dict_in[regu][sde]["spectra_MSE"][version] = None
                dict_in[regu][sde]["spectra_log_MSE"][version] = None

                dict_in[regu][sde]["MSE"][version] = None
                dict_in[regu][sde]["pearson"][version] = None
                dict_in[regu][sde]["spearman"][version] = None
                dict_in[regu][sde]["time_mean"][version] = None
                dict_in[regu][sde]["time_std"][version] = None
                dict_in[regu][sde]["spatial_mean"][version] = None
                dict_in[regu][sde]["spatial_std"][version] = None
                dict_in[regu][sde]["spatial_std_mag_U"][version] = None
                dict_in[regu][sde]["mag_U_fft"][version] = None


                dict_in[regu][sde]["curl"][version] = None
                dict_in[regu][sde]["q_crit"][version] = None
                dict_in[regu][sde]["time_histogram"][version] = None
                dict_in[regu][sde]["spatial_histogram"][version] = None
                dict_in[regu][sde]["curl_spatial_histogram"][version] = None
                dict_in[regu][sde]["q_crit_spatial_histogram"][version] = None
                dict_in[regu][sde]["wasserstein_distance"][version] = None
                dict_in[regu][sde]["KL_divergence"][version] = None
                dict_in[regu][sde]["wasserstein_distance_q_crit"][version] = None
                dict_in[regu][sde]["KL_divergence_q_crit"][version] = None
                dict_in[regu][sde]["wasserstein_distance_curl"][version] = None
                dict_in[regu][sde]["KL_divergence_curl"][version] = None
            except:
                dict_in[regu][sde]["u"] = {}
                dict_in[regu][sde]["v"] = {}
                dict_in[regu][sde]["w"] = {}
                dict_in[regu][sde]["Bx"] = {}
                dict_in[regu][sde]["By"] = {}
                dict_in[regu][sde]["Bz"] = {}
                dict_in[regu][sde]["mag_U"] = {}
                dict_in[regu][sde]["mag_B"] = {}

                dict_in[regu][sde]["RE"] = {}
                dict_in[regu][sde]["RE_U"] = {}
                dict_in[regu][sde]["RE_B"] = {}
                dict_in[regu][sde]["RE_mag_U"] = {}
                dict_in[regu][sde]["RE_mag_B"] = {}

                dict_in[regu][sde]["spectra"] = {}
                dict_in[regu][sde]["mag_spectra"] = {}

                dict_in[regu][sde]["spectra_MSE"] = {}
                dict_in[regu][sde]["spectra_log_MSE"] = {}

                dict_in[regu][sde]["MSE"] = {}
                dict_in[regu][sde]["pearson"] = {}
                dict_in[regu][sde]["spearman"] = {}
                dict_in[regu][sde]["time_mean"] = {}
                dict_in[regu][sde]["time_std"] = {}
                dict_in[regu][sde]["spatial_mean"] = {}
                dict_in[regu][sde]["spatial_std"] = {}
                dict_in[regu][sde]["spatial_std_mag_U"] = {}
                dict_in[regu][sde]["mag_U_fft"] = {}

                dict_in[regu][sde]["curl"] = {}
                dict_in[regu][sde]["q_crit"] = {}
                dict_in[regu][sde]["time_histogram"] = {}
                dict_in[regu][sde]["spatial_histogram"] = {}
                dict_in[regu][sde]["curl_spatial_histogram"] = {}
                dict_in[regu][sde]["q_crit_spatial_histogram"] = {}
                dict_in[regu][sde]["wasserstein_distance"] = {}
                dict_in[regu][sde]["KL_divergence"] = {}
                dict_in[regu][sde]["wasserstein_distance_q_crit"] = {}
                dict_in[regu][sde]["KL_divergence_q_crit"] = {}
                dict_in[regu][sde]["wasserstein_distance_curl"] = {}
                dict_in[regu][sde]["KL_divergence_curl"] = {}

            if perona:
                if dataset_name == "JHTDB":
                    (dict_in[regu][sde]["pred"][version][:, 0],
                     dict_in[regu][sde]["pred"][version][:, 1]) = (
                        smooth_flow_anisotropic(dict_in[regu][sde]["pred"][version][:, 0],
                                                dict_in[regu][sde]["pred"][version][:, 1],
                                                num_iters=50, kappa=0.05, step=0.03))
                elif dataset_name == "turbulent_radiative_layer_2D":
                    (dict_in[regu][sde]["pred"][version][:, 2],
                     dict_in[regu][sde]["pred"][version][:, 3]) = (
                        smooth_flow_anisotropic(dict_in[regu][sde]["pred"][version][:, 0],
                                                dict_in[regu][sde]["pred"][version][:, 1],
                                                num_iters=10, kappa=0.25, step=0.1))
                else:
                    raise NotImplementedError

            # --- velocity extraction ---
            if dataset_name == "JHTDB":
                u = dict_in[regu][sde]["pred"][version][:, 0]
                v = dict_in[regu][sde]["pred"][version][:, 1]
            elif dataset_name == "turbulent_radiative_layer_2D":
                u = dict_in[regu][sde]["pred"][version][:, 2]
                v = dict_in[regu][sde]["pred"][version][:, 3]
            elif dataset_name == "MHD_64":
                u = dict_in[regu][sde]["pred"][version][:, -3]
                v = dict_in[regu][sde]["pred"][version][:, -2]
                w = dict_in[regu][sde]["pred"][version][:, -1]

                Bx = dict_in[regu][sde]["pred"][version][:, 1]
                By = dict_in[regu][sde]["pred"][version][:, 2]
                Bz = dict_in[regu][sde]["pred"][version][:, 3]
            else:
                raise ValueError("Unknown dataset name")

            if dim == 2:
                dict_in[regu][sde]["u"] = copy.deepcopy(u)
                dict_in[regu][sde]["v"] = copy.deepcopy(v)
                dict_in[regu][sde]["spectra"][version] = Energy_spectra(u, v)
                dict_in[regu][sde]["mag_U"][version] = np.sqrt(np.square(u) + np.square(v))


                U_pred, U = [dict_in["GT"]["u"], dict_in["GT"]["v"]], [u, v]
                dict_in[regu][sde]["RE_U"][version] = dict_in[regu][sde]
                dict_in[regu][sde]["RE_mag_U"][version] = compute_relative_error(dict_in["GT"]["mag_U"], dict_in[regu][sde]["mag_U"][version])

            else:
                dict_in[regu][sde]["spectra"][version] = Energy_spectra(u, v, w)
                dict_in[regu][sde]["mag_U"][version] = np.sqrt(np.square(u) + np.square(v) + np.square(w))

                U_pred, U = [dict_in["GT"]["u"], dict_in["GT"]["v"], dict_in["GT"]["w"]], [u, v, w]
                dict_in[regu][sde]["RE_U"][version] = compute_relative_error(np.stack(U_pred, axis=1), np.stack(U, axis=1))
                dict_in[regu][sde]["RE_mag_U"][version] = compute_relative_error(dict_in["GT"]["mag_U"], dict_in[regu][sde]["mag_U"][version])

                if dataset_name == "MHD_64":
                    dict_in[regu][sde]["Bx"] = copy.deepcopy(Bx)
                    dict_in[regu][sde]["By"] = copy.deepcopy(By)
                    dict_in[regu][sde]["Bz"] = copy.deepcopy(Bz)

                    dict_in[regu][sde]["mag_spectra"][version] = Energy_spectra(Bx, By, Bz)  # Spectra magnetic field
                    dict_in[regu][sde]["mag_B"][version] = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))  # Magnitude B

                    B_pred, B = [dict_in["GT"]["Bx"], dict_in["GT"]["By"], dict_in["GT"]["Bz"]], [Bx, By, Bz]
                    dict_in[regu][sde]["RE_B"][version] = compute_relative_error(np.stack(B_pred, axis=1), np.stack(B, axis=1))
                    dict_in[regu][sde]["RE_mag_B"][version] = compute_relative_error(dict_in["GT"]["mag_B"], dict_in[regu][sde]["mag_B"][version])

            dict_in[regu][sde]["RE"][version] = {}
            for field in range(len(fields_name)):
                dict_in[regu][sde]["RE"][version][field] = compute_relative_error(dict_in["GT"]["pred"][:,field,...],
                                                                                  dict_in[regu][sde]["pred"][version][:,field,...])

            if dataset_name == "JHTDB":
                min_ = 2**2*1.1
                max_ = 2**6
            if dataset_name == "turbulent_radiative_layer_2D":
                min_ = 2**2
                max_ = 2**6*1.5
            if dataset_name == "MHD_64":
                min_ = 2**0,
                max_ = 2**4*1.5

            k_vals = dict_in[regu][sde]["spectra"][version]["k"]
            mask = (k_vals >= min_) & (k_vals <= max_)

            gt_E_k = dict_in["GT"]["spectra"]["E_k"][mask]
            pred_E_k = dict_in[regu][sde]["spectra"][version]["E_k"][mask]

            dict_in[regu][sde]["spectra_MSE"][version] = np.mean(np.square(gt_E_k - pred_E_k))
            dict_in[regu][sde]["spectra_log_MSE"][version] = np.mean(np.square(np.log10(gt_E_k + 1e-12) - np.log10(pred_E_k + 1e-12)))

            # dict_in[regu][sde]["mag_U_fft"][version] = compute_mag_U_frequency_series(dict_in[regu][sde]["mag_U"][version])

            if dim == 2:
                dict_in[regu][sde]["MSE"][version] = np.mean(np.square(dict_in["GT"]["pred"] - dict_in[regu][sde]["pred"][version]), axis=(2,3))
            else:
                dict_in[regu][sde]["MSE"][version] = np.mean(np.square(dict_in["GT"]["pred"] - dict_in[regu][sde]["pred"][version]), axis=(2,3,4))

            dict_in[regu][sde]["pearson"][version] = pearson_correlation_over_time(dict_in["GT"]["pred"], dict_in[regu][sde]["pred"][version])
            # dict_in[regu][sde]["spearman"][version] = spearman_correlation_over_time(dict_in["GT"]["pred"], dict_in[regu][sde]["pred"][version])


            # --- stats ---
            dict_in[regu][sde]["time_mean"][version] = np.mean(dict_in[regu][sde]["pred"][version], axis=0)
            dict_in[regu][sde]["time_std"][version] = np.std(dict_in[regu][sde]["pred"][version], axis=0)

            if dim == 2:
                dict_in[regu][sde]["spatial_mean"][version] = np.mean(dict_in[regu][sde]["pred"][version], axis=(2,3))
                dict_in[regu][sde]["spatial_std"][version] = np.std(dict_in[regu][sde]["pred"][version], axis=(2,3))
                dict_in[regu][sde]["spatial_std_mag_U"][version] = np.std(dict_in[regu][sde]["mag_U"][version], axis=(1,2))


                # --- phi properties ---
                dict_in[regu][sde]["curl"][version] = compute_2Dcurl(u, v, dx=1., dy=1.)
                dict_in[regu][sde]["q_crit"][version] = compute_2Dq_criterion(u, v, dx=1., dy=1.)
            else:
                dict_in[regu][sde]["spatial_mean"][version] = np.mean(dict_in[regu][sde]["pred"][version], axis=(2,3,4))
                dict_in[regu][sde]["spatial_std"][version] = np.std(dict_in[regu][sde]["pred"][version], axis=(2,3,4))
                dict_in[regu][sde]["spatial_std_mag_U"][version] = np.std(dict_in[regu][sde]["mag_U"][version], axis=(1,2,3))


                # --- phi properties ---
                dict_in[regu][sde]["curl"][version] = compute_3Dcurl(u, v, w, dx=1., dy=1., dz=1.)
                dict_in[regu][sde]["q_crit"][version] = compute_3Dq_criterion(u, v, w, dx=1., dy=1., dz=1.)

            # --- per field histogram ---
            dict_in[regu][sde]["time_histogram"][version] = {}
            dict_in[regu][sde]["spatial_histogram"][version] = {}
            dict_in[regu][sde]["wasserstein_distance"][version] = {}
            dict_in[regu][sde]["KL_divergence"][version] = {}

            for field in range(len(fields_name)):
                pred_field = dict_in[regu][sde]["pred"][version][:,field]
                if dim == 2:
                    T, Lx, Ly = pred_field.shape
                else:
                    T, Lx, Ly, Lz = pred_field.shape

                bin_edges_space = dict_in["GT"]["support"][field]
                histKwargs = dict(bins=bin_edges_space, density=True)
                # dict_in[regu][sde]["time_histogram"][version][field] = np.zeros((Lx, Ly, n_bins), dtype=np.int64)
                dict_in[regu][sde]["spatial_histogram"][version][field] = np.zeros((T, n_bins), dtype=np.int64)
                dict_in[regu][sde]["wasserstein_distance"][version][field] = np.zeros((T))
                dict_in[regu][sde]["KL_divergence"][version][field] = np.zeros((T))

                # for ix in range(Lx):  # Time histo, (Lx, Ly, support) -> plot(
                #     for iy in range(Ly):
                #         time_series = pred_field[:, ix, iy]
                #         hist_counts, _ = np.histogram(time_series, **histKwargs)
                #         dict_in[regu][sde]["time_histogram"][version][field][ix, iy, :] = hist_counts
                for t in range(T):  # Spatial histo, (t, support) -> plot(support, t=x) observe field's average value at a given time
                    spatial_values = pred_field[t].ravel()
                    hist_counts, _ = np.histogram(spatial_values, **histKwargs)
                    dict_in[regu][sde]["spatial_histogram"][version][field][t, :] = hist_counts

                    dict_in[regu][sde]["wasserstein_distance"][version][field][t] = (
                        wasserstein_distance(dict_in["GT"]["spatial_histogram"][field][t, :],
                                             dict_in[regu][sde]["spatial_histogram"][version][field][t, :]
                                             )
                    )
                    dict_in[regu][sde]["KL_divergence"][version][field][t] = (
                        kl_divergence(dict_in["GT"]["spatial_histogram"][field][t, :],
                                             dict_in[regu][sde]["spatial_histogram"][version][field][t, :]
                                             )
                    )

            q_crit = dict_in[regu][sde]["q_crit"][version]
            abs_curl = np.abs(dict_in[regu][sde]["curl"][version])
            if dim == 2:
                T, Lx, Ly = q_crit.shape
            else:
                T, Lx, Ly, Lz = q_crit.shape

            bins_q_crit = dict_in["GT"]["q_crit_support"]
            bins_curl = dict_in["GT"]["curl_support"]

            dict_in[regu][sde]["curl_spatial_histogram"][version] = np.zeros((T, n_bins), dtype=np.int64)
            dict_in[regu][sde]["q_crit_spatial_histogram"][version] = np.zeros((T, n_bins), dtype=np.int64)

            dict_in[regu][sde]["wasserstein_distance_curl"][version] = np.zeros((T))
            dict_in[regu][sde]["KL_divergence_curl"][version] = np.zeros((T))
            dict_in[regu][sde]["wasserstein_distance_q_crit"][version] = np.zeros((T))
            dict_in[regu][sde]["KL_divergence_q_crit"][version] = np.zeros((T))

            for t in range(T):
                hist_q_crit, _ = np.histogram(q_crit[t].ravel(), bins=bins_q_crit, density=True)

                dict_in[regu][sde]["q_crit_spatial_histogram"][version][t, :] = hist_q_crit

                if dim == 2:
                    hist_curl, _ = np.histogram(abs_curl[t].ravel(), bins=bins_curl, density=True)
                    dict_in[regu][sde]["curl_spatial_histogram"][version][t, :] = hist_curl



                dict_in[regu][sde]["wasserstein_distance_curl"][version][t] = (
                    wasserstein_distance(dict_in["GT"]["curl_spatial_histogram"][t, :],
                                         dict_in[regu][sde]["curl_spatial_histogram"][version][t, :]
                                         )
                )
                dict_in[regu][sde]["KL_divergence_curl"][version][t] = (
                    kl_divergence(dict_in["GT"]["curl_spatial_histogram"][t, :],
                                  dict_in[regu][sde]["curl_spatial_histogram"][version][t, :]
                                  )
                )
                dict_in[regu][sde]["wasserstein_distance_q_crit"][version][t] = (
                    wasserstein_distance(dict_in["GT"]["q_crit_spatial_histogram"][t, :],
                                         dict_in[regu][sde]["q_crit_spatial_histogram"][version][t, :]
                                         )
                )
                dict_in[regu][sde]["KL_divergence_q_crit"][version][t] = (
                    kl_divergence(dict_in["GT"]["q_crit_spatial_histogram"][t, :],
                                  dict_in[regu][sde]["q_crit_spatial_histogram"][version][t, :]
                                  )
                )


