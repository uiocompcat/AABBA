import wandb
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

from tools import calculate_r_squared


def plot_target_histogram(train_true_values, val_true_values, test_true_values, file_path='./image.png'):

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_true_values, label='Train', alpha=0.5)
    ax.hist(val_true_values, label='Val', alpha=0.5,)
    ax.hist(test_true_values, label='Test', alpha=0.5,)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Frequency')
    ax.legend()

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_correlation(predicted_values: list, true_values: list, file_path='./image.png'):

    # cast lists to arrays
    predicted_values = np.array(predicted_values)
    true_values = np.array(true_values)

    # set up canvas
    fig, ax = plt.subplots(figsize=(5, 5))

    # taken from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
    # get interpolated point densities
    data, x_e, y_e = np.histogram2d(predicted_values, true_values, bins=20, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([predicted_values, true_values]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    predicted_values, true_values, z = predicted_values[idx], true_values[idx], z[idx]

    # set base points with density coloring
    ax.scatter(predicted_values, true_values, c=z, cmap='Blues')

    # get min and max values
    min_value = min(predicted_values)
    max_value = max(predicted_values)

    # regression line
    z = np.polyfit(predicted_values, true_values, 1)
    p = np.poly1d(z)
    ax.plot([min_value, max_value], [p(min_value), p(max_value)], "r--")

    # formatting
    ax.text(0.2, 0.9, 'R² = ' + str(np.round(calculate_r_squared(np.array(predicted_values), np.array(true_values)), decimals=3)), size=10, color='blue', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('True values')

    # set same length axis ranges with 5% margin of max value
    ax.set_xlim([min_value - 0.05 * max_value, max_value + 0.05 * max_value])
    ax.set_ylim([min_value - 0.05 * max_value, max_value + 0.05 * max_value])

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
