import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def set_font_size_figures(fontsize):
    """Sets fontsize to figures.
    """
    params = {'legend.fontsize': fontsize,
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

def barplot(df, colors, labels, alpha, figname):
    """Makes a barplot of top 20 features.
    Arguments
        df,                 pd.DataFrame
            Containing the relevance data.
        colors,             list
            List of colors of length 20 specifying each
            barplot color.
        labels,             list
            List of length 2, where the first index contains the
            x-label and the second index contains the y-label.
        alpha,              float
            Transparency of barplots.
        figname,            string
            String specifying saving path of figure.

    """
    pos = np.arange(20) + 0.5
    fig = plt.figure()
    plt.bar(pos,
            100*df['relevance'][0:20],
            color = colors,
            yerr = 100*df['relevance_seotm'][0:20],
            linewidth = 1,
            edgecolor = 'white',
            alpha = alpha)
    plt.xticks(pos, df["feature"][0:20])
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.xticks(rotation = 'vertical')
    plt.savefig(figname, format='pdf', bbox_inches = 'tight')


def error_plot(x, y, yerr, labels, colors, full, fullerr, name):
    """
    Arguments
        x,          array-like
            X-axis values.
        y,          array-like
            Y-axis values.
        yerr,       array-like
            Error in y-values.
        labels,     list
            List with 0th index is xlabel, 1st index is ylabel.
        colors,     list
            List with 0th index is color of errorbars,
            1st index is color of original input,
            2nd index is color of 95% CI of original input.
        full,       float
            Value for non-reduced input (full ACs).
        fullerr,    float
            Error for non-reduced input (full ACs).
        name,       string
            Path to save to.

    """
    fig = plt.figure()
    plt.errorbar(x, y, yerr,
                 fmt = "s",
                 label = "95% CI reduced input",
                 color = colors[0])
    plt.hlines(full,
               min(x),
               max(x),
               colors = [colors[1]],
               linestyles = ["dashed"],
               label = "Original input")
    ax = plt.gca()
    ax.fill_between(x,
                    [full + fullerr for i in range(len(x))],
                    [full - fullerr for i in range(len(x))],
                    color = colors[2], alpha = 0.7,
                    label = "95% CI original input")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.savefig(name, format="pdf", bbox_inches = "tight")

def scatter_plot(x, y, labels, colors, full, name):
    fig = plt.figure()
    plt.plot(x, y, 'o',
             label = "Reduced input",
             color = colors[0])
    plt.hlines(full, min(x), max(x),
               colors = [colors[1]],
               linestyles = ["dashed"],
               label = "Original input")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.savefig(name, format = "pdf", bbox_inches = "tight")
