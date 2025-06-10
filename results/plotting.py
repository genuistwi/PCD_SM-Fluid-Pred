from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


# A color map that is duplicated in reverse so twice the range
def looped_cmap(cmap="jet"):
    jet = plt.get_cmap(cmap, 256)
    colors = jet(np.linspace(0, 1, 256))
    looped_colors = np.vstack((colors, colors[::-1]))
    looped_cm = LinearSegmentedColormap.from_list('looped', looped_colors)
    return looped_cm

def emphasized_cmap(cmap='Spectral'):
    jet = plt.get_cmap(cmap)

    # Define a non-linear mapping to emphasize higher values
    def emphasize_high_values(x):
        return x**2. # sqrt gives more resolution at high end (x closer to 1)

    n = 256
    colors = jet(emphasize_high_values(np.linspace(0, 1, n)))
    custom_jet_high_detail = LinearSegmentedColormap.from_list('custom_jet_high_detail', colors, N=n)

    return custom_jet_high_detail



def compa_animation(GT, pred, fields_name, T_lim, cmap, vmin, vmax,
                    ratio_xy, coeff,
                    n_fields, cbarKwargs, adjustKwargs, cbarLoc,
                    Y_axisKwargs, titlesKwargs, suptitleKwargs):

    """
    n_fields by 2 plotting of 2D images. Send back an animation
    """

    fig, axes = plt.subplots(2, n_fields, figsize=(coeff * n_fields, coeff * ratio_xy * 2))
    [(ax.set_xticks([]), ax.set_yticks([])) for ax in axes.flat]  # Remove axes


    # --- Titles and labels ---
    title_string = 'Predictions for each field, '  # Main title
    suptitle = plt.suptitle(title_string + r"$\tau=0$", **suptitleKwargs)
    axes[1, 0].set_ylabel("Prediction", **Y_axisKwargs)
    axes[0, 0].set_ylabel('Ground truth', **Y_axisKwargs)


    img_format, rows_frame = [], []
    for j in range(n_fields):  # GT
        im = axes[0, j].imshow(GT[0, j, :, :], vmin=vmin[j], vmax=vmax[j], cmap=cmap)
        axes[0, j].set_title(fields_name[j], **titlesKwargs); rows_frame.append(im)

    img_format.append(rows_frame); rows_frame = []
    for j in range(n_fields):  # Pred
        im = axes[len(img_format), j].imshow(pred[0, j, :, :], vmin=vmin[j], vmax=vmax[j], cmap=cmap)
        rows_frame.append(im)
    img_format.append(rows_frame)

    # cbar.set_label("Color Scale")

    def update(frame):
        suptitle.set_text(title_string + r"$\tau=$" + f"{frame}")
        for j in range(n_fields): img_format[0][j].set_data(GT[frame, j, :, :])
        for j in range(n_fields): img_format[1][j].set_data(pred[frame, j, :, :])

        return [im for row in img_format for im in row]


    ticks = np.linspace(vmin[-1], vmax[-1], 5)
    tick_labels = [f"{t:.1f}" for t in ticks]

    cbar_ax = fig.add_axes(cbarLoc)  # adjust to taste
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=ticks, **cbarKwargs)
    cbar.set_ticklabels([])  # cheat by replacing the label but not distort the cbar

    plt.subplots_adjust(**adjustKwargs)

    return animation.FuncAnimation(fig, update, frames=T_lim, interval=100, blit=True)


def compa_time_series(time_series, titles, x_labels, y_labels,
                      titleKwargs, x_labelsKwargs, y_labelsKwargs, cmap, coeff, vmin, vmax, ratio_xy, cbarKwargs,
                      suptitleKwargs, cbar_labels, cbarLabelKwargs={},
                      rgba=None, suptitle=None, save_string=None, dpi=200, adjustKwargs=None):

    """
    multiple field plot/time series. With X legend
    """

    # In: [num_series, n_frames, Lx, Ly]
    # vmin/vmax is list of CB val by frame
    n_series = time_series.shape[0]
    n_frames = time_series.shape[1]

    fig, axes = plt.subplots(n_series, n_frames, figsize=(coeff * n_frames, coeff * ratio_xy * n_series))
    [(ax.set_xticks([]), ax.set_yticks([])) for ax in axes.flat]  # Remove axes

    for i in range(n_series):
        for j in range(n_frames):
            ax = axes[i, j]
            im = ax.imshow(time_series[i, j, ...], vmin=vmin[j], vmax=vmax[j], cmap=cmap)
            if i == 0:
                ax.set_title(titles[j], **titleKwargs)
            if x_labels[i, j] is not None:
                ax.set_xlabel(x_labels[i, j], **x_labelsKwargs)
            if j == 0:
                ax.set_ylabel(y_labels[i], **y_labelsKwargs)

    ticks = np.linspace(vmin[-1], vmax[-1], 5)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', **cbarKwargs, ticks=ticks, aspect=50)
    cbar.set_ticklabels(cbar_labels, **cbarLabelKwargs)

    if rgba is not None:
        for ax in axes.flat:
            ax.imshow(rgba, interpolation='none', zorder=100)

    if suptitle is not None: plt.suptitle(suptitle, **suptitleKwargs)
    if adjustKwargs is not None: plt.subplots_adjust(**adjustKwargs)

    plt.savefig(save_string, dpi=dpi, bbox_inches='tight')

    plt.show()

def compa_time_series_err(time_series, titles, x_labels, y_labels,
                      titleKwargs, x_labelsKwargs, y_labelsKwargs, cmap, coeff, vmin, vmax, ratio_xy, cbarKwargs,
                      suptitleKwargs,
                      rgba=None, suptitle=None, save_string=None, dpi=200, adjustKwargs=None):


    # In: [num_series, n_frames, Lx, Ly]
    # vmin/vmax is list of CB val by frame
    n_series = time_series.shape[0]
    n_frames = time_series.shape[1]

    fig, axes = plt.subplots(n_series, n_frames, figsize=(coeff * n_frames, coeff * ratio_xy * n_series))
    [(ax.set_xticks([]), ax.set_yticks([])) for ax in axes.flat]  # Remove axes

    for i in range(n_series):
        for j in range(n_frames):
            ax = axes[i, j]

            if i == 2 or i == 4:
                im = ax.imshow(time_series[i, j, ...], vmin=0, vmax=1, cmap="seismic")
            else:
                im = ax.imshow(time_series[i, j, ...], vmin=vmin[j], vmax=vmax[j], cmap=cmap)

            if i == 0:
                ax.set_title(titles[j], **titleKwargs)
            if x_labels[i, j] is not None:
                ax.set_xlabel(x_labels[i, j], **x_labelsKwargs)
            if j == 0:
                ax.set_ylabel(y_labels[i], **y_labelsKwargs)

    ticks = np.linspace(0,1, 5)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', **cbarKwargs, ticks=ticks, aspect=50)
    cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
    cbar.set_label('Rel. error', **x_labelsKwargs)

    if rgba is not None:
        for ax in axes.flat: ax.imshow(rgba, interpolation='none', zorder=100)

    if suptitle is not None: plt.suptitle(suptitle, **suptitleKwargs)

    if adjustKwargs is not None: plt.subplots_adjust(**adjustKwargs)


    plt.savefig(save_string, dpi=dpi, bbox_inches='tight')

    plt.show()

def compa_time_series_err_2(time_series, titles, x_labels, y_labels,
                      titleKwargs, x_labelsKwargs, y_labelsKwargs, cmap, coeff, vmin, vmax, ratio_xy, cbarKwargs,
                      suptitleKwargs, left_cbar_args, top_cbar_args, top_label_size, neg=False,
                      rgba=None, suptitle=None, save_string=None, dpi=200, adjustKwargs=None):


    n_series = time_series.shape[0]
    n_frames = time_series.shape[1]

    fig_height = coeff * ratio_xy * (n_series + 0.3)  # Extra height for colorbar row
    fig_width = coeff * n_frames
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid with extra row for colorbars
    if adjustKwargs is not None:
        gs = gridspec.GridSpec(n_series + 1, n_frames, height_ratios=[1]*n_series + [0.05], **adjustKwargs)
    else:
        gs = gridspec.GridSpec(n_series + 1, n_frames, height_ratios=[1]*n_series + [0.05])


    axes = np.empty((n_series, n_frames), dtype=object)
    ims = np.empty((n_series, n_frames), dtype=object)

    # Plot data fields
    for i in range(n_series):
        for j in range(n_frames):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 2 or i == 4:
                im = ax.imshow(time_series[i, j, ...], vmin=0, vmax=1, cmap="seismic")
            else:
                im = ax.imshow(time_series[i, j, ...], vmin=vmin[j], vmax=vmax[j], cmap=cmap)

            ims[i, j] = im

            if i == 0:
                ax.set_title(titles[j], pad=15, **titleKwargs)
            if x_labels[i, j] is not None:
                ax.set_xlabel(x_labels[i, j], **x_labelsKwargs)
            if j == 0:
                ax.set_ylabel(y_labels[i], **y_labelsKwargs)

    for j in range(n_frames):
        ax_data = axes[0, j]  # top row axis of column j
        im = ims[0, j]        # matching image handle

        # Place colorbar inset *above* the top-row plot
        cax = ax_data.inset_axes(top_cbar_args)
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 5)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks, **cbarKwargs)
        cbar.set_ticklabels([f"{v:.2f}" for v in ticks])
        cbar.ax.tick_params(labelsize=top_label_size)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")

    # Optional overlay
    if rgba is not None:
        for ax in axes.flat:
            ax.imshow(rgba, interpolation='none', zorder=100)

    # Optional suptitle
    if suptitle is not None:
        plt.suptitle(suptitle, **suptitleKwargs)

    im = ims[-1, -1]
    cbar_ax = fig.add_axes(left_cbar_args)  # adjust these numbers if needed
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=np.linspace(im.get_clim()[0], im.get_clim()[1], 5))
    cbar.set_label("Rel. err", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Save and display
    plt.savefig(save_string, dpi=dpi, bbox_inches='tight')
    plt.show()



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def _flatten(d):
    """Return a 1‑D np.array no matter if *d* is list/array or dict."""
    if isinstance(d, dict):
        return np.concatenate([np.ravel(v) for v in d.values()])
    return np.ravel(d)


def _mean_var(d, quart):
    d = _flatten(d)
    if quart:
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        return q1, q3
    else:
        return d.mean(), d.var(ddof=1)

def _scientific_exponent(vals):
    mags = [abs(v) for v in vals if v != 0]
    if not mags:
        return 0
    exp = int(math.floor(math.log10(max(mags))))
    # keep mantissas in [0.01,100)
    if max(mags) / 10**exp < 0.1:
        exp -= 1
    return exp

def build_latex_table(
        test_cases,
        sde_names,                     # iterable
        regularization,                # iterable of True/False (or anything hashable)
        measures,                      # iterable of keys
        measure_labels,                # len == len(measures)
        versions,                      # iterable
        *,
        sde_labels=None,               # mapping or iterable aligned w/ sde_names
        regu_labels=None,              # mapping or iterable aligned w/ regularization
        highlight_lowest=None,         # list[bool] same length as measures
        num_decimals=2,
        pms,                        # +- var display list
        quart=False
):

    # ------- label helpers -------
    if sde_labels is None:
        sde_labels = {k: str(k) for k in sde_names}
    elif not isinstance(sde_labels, dict):
        sde_labels = {k: lbl for k, lbl in zip(sde_names, sde_labels)}

    if regu_labels is None:
        regu_labels = {k: str(k) for k in regularization}
    elif not isinstance(regu_labels, dict):
        regu_labels = {k: lbl for k, lbl in zip(regularization, regu_labels)}

    if highlight_lowest is None:
        highlight_lowest = [True] * len(measures)

    # ---------- aggregation ----------
    result = {}
    for reg, sde in product(regularization, sde_names):
        rows = []
        for m in measures:
            cols = []
            for tc in test_cases:
                data = [_flatten(tc[reg][sde][m][v]) for v in versions]
                data = np.concatenate(data)
                cols.append(_mean_var(data, quart))
            rows.append(cols)
        result[(reg, sde)] = rows

    # ---------- exponents per measure ----------
    exponents = []
    for mi, m in enumerate(measures):
        vals = [result[(r, s)][mi][ti][0]
                for r in regularization for s in sde_names
                for ti in range(len(test_cases))]
        exponents.append(_scientific_exponent(vals))

    # ---------- collect scaled means (for boldface) ----------
    n_cols = len(test_cases) * len(measures)
    column_values = [[] for _ in range(n_cols)]
    for reg, sde in product(regularization, sde_names):
        for mi in range(len(measures)):
            exp = exponents[mi]
            for ti in range(len(test_cases)):
                mean = result[(reg, sde)][mi][ti][0] / 10**exp
                column_values[mi*len(test_cases)+ti].append(mean)

    # choose minima or maxima per measure
    column_best = []
    for mi in range(len(measures)):
        for ti in range(len(test_cases)):
            col_idx = mi*len(test_cases)+ti
            if highlight_lowest[mi]:
                column_best.append(min(column_values[col_idx]))
            else:
                column_best.append(max(column_values[col_idx]))

    # ---------- LaTeX assembly ----------
    def fmt(mean, var, exp, pm):
        mean_s = f"{mean/10**exp:.{num_decimals}f}"
        var_s  = f"{var/10**exp:.{num_decimals}f}"

        # if (var/10**exp) > (10*mean/10**exp):
        #     var_s = "\,k\,"

        return f"{mean_s} $\\pm$ {var_s}" if pm else f"{mean_s}"

    col_format = "|c|c|" + "|".join("c"*len(test_cases) for _ in measures) + "|"

    lines = [f"\\begin{{tabular}}{{{col_format}}}",
             "\\hline",
             "\\textbf{SDE type} & \\textbf{Regularization} & " +
             " & ".join(
                 f"\\multicolumn{{{len(test_cases)}}}{{c|}}{{{lbl} ($10^{{{exp}}}$)}}"
                 for lbl, exp in zip(measure_labels, exponents)
             ) + " \\\\"]

    # clines (first block already done)
    running = 3
    for i in range(len(measures)):
        start = running
        end   = running + len(test_cases) - 1
        lines.append(f"\\cline{{{start}-{end}}}")
        running += len(test_cases)

    # table body
    for reg in regularization:
        for sde in sde_names:
            row_cells = []
            for mi in range(len(measures)):
                pm = pms[mi]
                exp = exponents[mi]
                for ti in range(len(test_cases)):
                    mean, var = result[(reg, sde)][mi][ti]
                    scaled_mean = mean / 10**exp
                    col_idx = mi*len(test_cases)+ti
                    cell = fmt(mean, var, exp, pm)
                    if math.isclose(scaled_mean, column_best[col_idx]):
                        cell = f"\\textbf{{{cell}}}"
                    row_cells.append(cell)

            lines.append(f"{sde_labels[sde]} & {regu_labels[reg]} & "
                         + " & ".join(row_cells) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    return "\n".join(lines)

