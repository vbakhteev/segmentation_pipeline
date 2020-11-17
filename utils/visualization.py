import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def metrics_to_image(metrics: dict, n_row=2) -> np.array:
    n_col = math.ceil(len(metrics) / n_row)

    fig, axes = plt.subplots(n_col, n_row)
    canvas = FigureCanvas(fig)

    for i, (name, metric_list) in enumerate(metrics.items()):
        ax = axes[i // n_row][i % n_col]

        x = range(1, len(metric_list) + 1)
        ax.plot(x, metric_list)
        ax.set_title(name)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))

    return image