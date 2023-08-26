import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting_functions import error_plot, set_font_size_figures
from data_preprocessing import find_accumulated_relevance



# set font size to figures
set_font_size_figures("x-large")
# define colors
red = "#B01729"; orange = "#F4604D"; beige = "#F4A581"
lightblue = "#8FC2DE"; blue = "#2066AB"; darkblue = "#042E61"

path = os.getcwd()
fig_path = path + "/results/MLP/figures/relevance_figures/"
data_path = path + "/results/MLP/data/"
inputs = ["periodic", "nbo"]
targets = ["barrier", "distance"]

for input_name in inputs:
    for target_name in targets:
        print(input_name + " " + target_name)
        df = pd.read_csv(data_path + "summary_" + input_name + "_" + target_name + ".csv")
        #print(df)
        x = df["nfeatures"].iloc[0:-1]
        y = df["test_mae"].iloc[0:-1]
        yerr = df["test_mae_stderr"].iloc[0:-1]
        full = df["test_mae"].iloc[-1]
        fullerr = df["test_mae_stderr"].iloc[-1]
        if target_name == "barrier":
            ylabel = r"MAE, $\Delta\mathrm{E}^\ddag$ [kcal/mol]"
        else:
            ylabel = r"MAE, d$_{\mathrm{H}\cdots\mathrm{H}}$ [Å]"
        labels = ["Number of features", ylabel]
        colors = [lightblue, red, orange]
        name = fig_path + "MLP_MAE_" + input_name + "_" + target_name + "_num_features.pdf"
        error_plot(x, y, 1.96*yerr, labels, colors, full, 1.96*fullerr, name)
        x = df["accumulated_relevance"].iloc[0:-1]
        labels = ["Accumulated relevance of input [%]", ylabel]
        name = fig_path + "MLP_MAE_" + input_name + "_" + target_name + "_relevance.pdf"
        error_plot(x, y, 1.96*yerr, labels, colors, full, 1.96*fullerr, name)
        y = df["test_r2"].iloc[0:-1]
        yerr = df["test_r2_stderr"].iloc[-1]
        full = df["test_r2"].iloc[-1]
        fullerr = df["test_r2"].iloc[-1]
        labels = ["Accumulated relevance of input [%]", r"r$^2$"]
        name = fig_path + "MLP_r2_" + input_name + "_" + target_name + "_relevance.pdf"
        error_plot(x, y, 1.96*yerr, labels, colors, full, 1.96*fullerr, name)
        x = df["nfeatures"].iloc[0:-1]
        labels = ["Number of features", r"r$^2$"]
        name = fig_path + "MLP_r2_" + input_name + "_" + target_name + "_num_features.pdf"
        error_plot(x, y, 1.96*yerr, labels, colors, full, 1.96*fullerr, name)
        idx_min = np.argmin(df["test_mae"])
        best_error = df["test_mae"].iloc[idx_min]; error_stderr = df["test_mae_stderr"].iloc[idx_min]
        best_r2 = df["test_r2"].iloc[idx_min]; r2_stderr = df["test_r2_stderr"].iloc[idx_min]
        best_nfeatures = df["nfeatures"].iloc[idx_min]; best_acc_rel = df["accumulated_relevance"].iloc[idx_min]
        one_run_best_mae = df["best_mae_test"].iloc[idx_min]; one_run_best_r2 = df["best_r2_test"].iloc[idx_min]

        print(f"nfeatures = {best_nfeatures:.0f}, accumulated relevance = {best_acc_rel}")
        print(f"MAE = {best_error:.4f}+-{1.96*error_stderr}, R2 = {best_r2:.4f}+-{1.96*r2_stderr}")
        print(f"Best: MAE = {one_run_best_mae}, R2 = {one_run_best_r2}")
        len_df = len(df["test_mae"])
        for i in range(len_df):
            print(f"nfeatures = {df['nfeatures'].iloc[i]}, accumulated_relevance = {df['accumulated_relevance'].iloc[i]}")
            print(f"MAE = {df['test_mae'].iloc[i]} +- {1.96*df['test_mae_stderr'].iloc[i]}")
            print(f"R2 = {df['test_r2'].iloc[i]} +- {1.96*df['test_r2_stderr'].iloc[i]}")
