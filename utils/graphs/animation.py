import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def multi_field_animation(dataObj, fields, num_run, fps, lim_frames=None):
    """Animate data over time, assuming dataObj sends objects 'F T Lx Ly'."""

    data = dataObj[num_run]
    numFields, time_steps, Lx, Ly = data.shape

    rows = (numFields + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2 * rows))
    axes = axes.flatten()

    # Min and max scales for each fields
    im_list = []
    vmin_list = [torch.min(data[i]) for i in range(numFields)]
    vmax_list = [torch.max(data[i]) for i in range(numFields)]

    # Init graph
    for i in range(numFields):
        im = axes[i].imshow(data[i, 0, :, :], animated=True, cmap='jet',
                            vmin=vmin_list[i], vmax=vmax_list[i])

        axes[i].set_title(fields[i])
        axes[i].set_xticks(np.round(np.linspace(0, Ly - 1, 2)))  # Ensure ticks start at origin
        axes[i].set_yticks(np.linspace(0, Lx - 1, 2))

        cbar = fig.colorbar(im, ax=axes[i], orientation='vertical', shrink=0.65, aspect=7)
        # cbar.set_label(f'Field {i + 1} Intensity')

        cbar_ticks = np.linspace(vmin_list[i], vmax_list[i], 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks])

        im_list.append(im)

    def update(frame):
        for i in range(numFields):
            im_list[i].set_array(data[i, frame, :, :])
        return im_list

    frames = time_steps if lim_frames is None else lim_frames
    ani = FuncAnimation(fig, update, blit=True, repeat=True, interval=1000 / fps, frames=frames, )  # frames=time_steps,)
    plt.tight_layout()
    plt.close()

    return ani
