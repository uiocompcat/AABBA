import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting_functions import scatter_plot, set_font_size_figures, barplot
from data_preprocessing import find_accumulated_relevance
"""
Colors used in Figure 3 in AABBA paper.
I thought these might be nice to use?
"""
red = "#B01729"; orange = "#F4604D"; beige = "#F4A581"
lightblue = "#8FC2DE"; blue = "#2066AB"; darkblue = "#042E61"

"""
-----------------------------------------------------------------------

                        Defining paths

-----------------------------------------------------------------------
"""
path = os.getcwd()
fig_path = path + "/results/feature_importance/figures/"
path_to_relevances = path + "/results/reduced_autocorrelation_vectors/"
"""
AABBA
"""
periodic_barrier = "periodic_relevance_barrier.csv"
periodic_distance = "periodic_relevance_distance.csv"
nbo_barrier = "nbo_relevance_barrier.csv"
nbo_distance = "nbo_relevance_distance.csv"
"""
AA only
"""
periodic_AA_barrier = "periodic_AA_relevance_barrier.csv"
periodic_AA_distance = "periodic_AA_relevance_distance.csv"
nbo_AA_barrier = "nbo_AA_relevance_barrier.csv"
nbo_AA_distance = "nbo_AA_relevance_distance.csv"
"""
BB only
"""
periodic_BB_barrier = "periodic_BB_relevance_barrier.csv"
periodic_BB_distance = "periodic_BB_relevance_distance.csv"
nbo_BB_barrier = "nbo_BB_relevance_barrier.csv"
nbo_BB_distance = "nob_BB_relevance_distance.csv"
"""
AB only
"""
periodic_AB_relevance_barrier = "periodic_AB_relevance_barrier.csv"
periodic_AB_relevance_distance = "periodic_AB_relevance_distance.csv"
nbo_AB_relevance_barrier = "nbo_AB_relevance_barrier.csv"
nbo_AB_relevance_distance = "nbo_AB_relevance_distance.csv"


def path_finder(input_, target):
    """
    Arguments
        input,              string
            periodic, periodic_AA, periodic_BB, periodic_AB
            nbo, nbo_AA, nbo_BB, or nbo_AB
        target,             string
            barrier or distance
    Returns
        path to .csv file
    """
    path = os.getcwd()
    path_to_relevances = path + "/results/reduced_autocorrelation_vectors/"
    name_of_csv = input_ + "_relevance_" + target + ".csv"
    return path_to_relevances + name_of_csv





# Setting font size
set_font_size_figures("x-large")
inputs = ["periodic", "nbo"]
targets = ["barrier", "distance"]
for input_ in inputs:
    for target in targets:
        df = pd.read_csv(path_finder(input_, target))
        colors_GP_barrier_suggestion = []
        for i in range(20):
            if "AA" in df["feature"][i]:
                colors_GP_barrier_suggestion.append(blue)
            elif "AB" in df["feature"][i]:
                colors_GP_barrier_suggestion.append(orange)
            else:
                colors_GP_barrier_suggestion.append(red)
        figname = fig_path + "feature_importance_" + input_ + "_" + target + ".pdf"
        xlabel = "AC descriptors"
        if target == "barrier":
            target_label = r"$\Delta\mathrm{E}^\ddag$"
        if target == "distance":
            target_label = r"d$_{\mathrm{H}\cdots\mathrm{H}}$"
        ylabel = r"AC descriptor relevance," + target_label + " [%]"

        barplot(df, colors_GP_barrier_suggestion,
                [xlabel, ylabel],
                1.0, figname)
