import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def metrics_to_image(metrics: dict, n_row=2) -> np.array:
    n_col = math.ceil(len(metrics) / n_row)

    fig, axes = plt.subplots(n_col, n_row)
    canvas = FigureCanvas(fig)

    for i, (name, metric_list) in enumerate(metrics.items()):
        if n_col == 1:
            ax = axes[i % n_row]
        else:
            ax = axes[i // n_row][i % n_row]

        x = range(1, len(metric_list) + 1)
        ax.plot(x, metric_list)
        ax.set_title(name)

    fig.tight_layout(pad=2.0)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))

    return image


def plot_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title("slice %d" % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap="gray")
        ax[int(i / rows), int(i % rows)].axis("off")
    plt.show()
