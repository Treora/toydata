from __future__ import division, print_function

from sys import stderr

import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np


def plot_correlations(correlations):
    """Plot the four correlation matrices, between the two layers of h and l.

       Note: This function is hard-coded to handle two layers of activations
       and two layers of labels (higher layers will just be ignored).
    """
    l1_size, h1_size = correlations[0][0].shape
    l2_size, h2_size = correlations[1][1].shape


    figure_size = _estimate_figure_size(l1_size+l2_size, h1_size+h2_size)
    fig = plt.figure(figsize=figure_size/80, dpi=80)
    gs = gridspec.GridSpec(2, 2, height_ratios=[l1_size, l2_size],
                           width_ratios=[h1_size, h2_size])
    for (i,j) in [(0,0), (0,1), (1,0), (1,1)]:
        ax = fig.add_subplot(gs[i,j])
        plot_matrix(correlations[i][j], scale=[-1, 1], ax=ax)
        ax.set_title("Correlations of l{0} with h{1}".format(i+1,j+1))
        ax.set_xlabel("h{}".format(j+1), fontsize='large')
        ax.set_ylabel("l{}".format(i+1), fontsize='large',
                      rotation='horizontal')
    fig.tight_layout()
    return fig


def plot_weights(weights):
    """Plot the given weight matrices. Pass them as [W_v_h1, W_h1_h2, ...]"""
    n = len(weights)
    output_sizes = [w.shape[0] for w in weights]
    input_sizes = [w.shape[1] for w in weights]
    labels = ['v'] + ['h{}'.format(i+1) for i in range(n)]

    figure_size = _estimate_figure_size(sum(output_sizes), max(input_sizes))
    fig = plt.figure(figsize=figure_size/80, dpi=80)

    # Create rows for the matrices, from bottom to top.
    rowgrid = gridspec.GridSpec(n, 1, height_ratios=output_sizes[::-1])

    # For each network layer, plot its weight matrix
    for i in range(n):
        # Start at the bottom row, so that highest layer ends up on top.
        row = rowgrid[n-1-i]

        padding = (max(input_sizes) - input_sizes[i]) / 2
        if padding != 0:
            # Add padding to make the plot width proportional to matrix width
            columngrid = gridspec.GridSpecFromSubplotSpec(1, 3,
                               subplot_spec=row,
                               width_ratios=[padding, input_sizes[i], padding])
            ax = fig.add_subplot(columngrid[1])
        else:
            ax = fig.add_subplot(row)

        # Plot the weight matrix into the subplot
        plot_matrix(weights[i], ax=ax)

        ax.set_title("Weight matrix of layer {}".format(i+1))
        ax.set_xlabel(labels[i], fontsize='large')
        ax.set_ylabel(labels[i+1], fontsize='large', rotation='horizontal')
    fig.tight_layout()
    return fig


def _estimate_figure_size(rows, columns):
    """Rough estimate of a reasonable figure size to fit matrices.
       Wish matplotlib would figure this out by itself.."""
    # Make column wide enough for ~4 characters each ~10 pixels wide, say 60px.
    # Same height for rows, to make cells squarish. And some extra space to
    # account for titles and labels and such.
    return np.array([columns * 60 + 50, rows * 60 + 50])


def plot_matrix(m, scale=None, ax=None, cmap=matplotlib.cm.coolwarm_r):
    """Display matrix m by showing numeric values with coloured backgrounds.

       Arguments:
       m: the matrix or 2d array to plot
       scale: the min and max for the colour map. Defaults to (min(m), max(m)).
       ax: the axes/subplot to plot into. Defaults to pyplot's current axis.
       cmap: colour map to use. Defaults to matplotlib.cm.coolwarm_r.
    """
    if scale is None:
        scale = [np.min(m), np.max(m)]
    rows, cols = m.shape

    if ax is None:
        ax = plt.gca() # get current axis

    # In the image, y will count up from bottom to top, so we flip the matrix
    m = m[::-1]

    # Create a colour plot
    ax.pcolor(m, vmin=scale[0], vmax=scale[1], cmap=cmap)

    # Write the indices below and besides the grid
    ax.set_yticks(np.arange(0.5, rows+0.5))
    ax.set_xticks(np.arange(0.5, cols+0.5))
    ax.set_yticklabels(range(rows-1, -1, -1), fontsize='large')
    ax.set_xticklabels(range(0, cols), fontsize='large')

    # Write the numeric value into the colour plot
    for x in range(cols):
        for y in range(rows):
            ax.annotate("{0:.2f}".format(m[y,x]), xy=(x+.5,y+.5), fontsize=15,
                horizontalalignment='center', verticalalignment='center')


def save_figure(fig, filename):
    try:
        fig.savefig(filename, transparent=True, bbox_inches='tight')
    except IOError as err:
        print('Could not write figure to file: ' + err.message, file=stderr)
