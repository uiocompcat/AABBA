import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting_functions import scatter_plot, set_font_size_figures
from data_preprocessing import find_accumulated_relevance



# set font size to figures
set_font_size_figures("x-large")
# define colors
red = "#B01729"; orange = "#F4604D"; beige = "#F4A581"
lightblue = "#8FC2DE"; blue = "#2066AB"; darkblue = "#042E61"

path = os.getcwd()
fig_path = path + "/results/GP/figures/"
data_path = path + "/results/GP/data/"
inputs = ["periodic", "nbo"]
targets = ["barrier", "distance"]

for input_name in inputs:
    for target_name in targets:
        print(input_name + " " + target_name)
        df = pd.read_csv(data_path + "summary_" + input_name + "_" + target_name + ".csv")
        x = df["nfeatures"].iloc[0:-1]
        y = df["test_mae"].iloc[0:-1]
        full = df["test_mae"].iloc[-1]
        if target_name == "barrier":
            ylabel = r"MAE, $\Delta\mathrm{E}^\ddag$ [kcal/mol]"
        else:
            ylabel = r"MAE, d$_{\mathrm{H}\cdots\mathrm{H}}$ [Å]"
        labels = ["Number of features", ylabel]
        colors = [lightblue, red]
        name = fig_path + "GP_MAE_" + input_name + "_" + target_name + "_num_features.pdf"
        scatter_plot(x, y, labels, colors, full, name)
        x = df["accumulated_relevance"].iloc[0:-1]
        labels = ["Accumulated relevance of input [%]", ylabel]
        name = fig_path + "GP_MAE_" + input_name + "_" + target_name + "_relevance.pdf"
        scatter_plot(x, y, labels, colors, full, name)
        y = df["test_r2"].iloc[0:-1]
        full = df["test_r2"].iloc[-1]
        labels = ["Accumulated relevance of input [%]", r"r$^2$"]
        name = fig_path + "GP_r2_" + input_name + "_" + target_name + "_relevance.pdf"
        scatter_plot(x, y, labels, colors, full, name)
        x = df["nfeatures"].iloc[0:-1]
        labels = ["Number of features", r"r$^2$"]
        name = fig_path + "GP_r2_" + input_name + "_" + target_name + "_num_features.pdf"
        scatter_plot(x, y, labels, colors, full, name)
        idx_min = np.argmin(df["test_mae"])
        best_error = df["test_mae"].iloc[idx_min]; best_r2 = df["test_r2"].iloc[idx_min]
        best_nfeatures = df["nfeatures"].iloc[idx_min]; best_acc_rel = df["accumulated_relevance"].iloc[idx_min]

        print("Best:")
        print(f"nfeatures = {best_nfeatures:.0f}, accumulated relevance = {best_acc_rel}")
        print(f"MAE = {best_error:.4f}, R2 = {best_r2:.4f}")
