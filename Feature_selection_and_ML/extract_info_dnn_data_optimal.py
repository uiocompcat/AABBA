import numpy as np
import pandas as pd
import os
from data_preprocessing import find_accumulated_relevance


def path_finder(input_, target):
    """Finds path to appropriate relevance data.
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


def extract_scores(df):
    idx = np.argmin(df["val_mae"])
    train_mae = df["train_mae"].iloc[idx]; val_mae = df["val_mae"].iloc[idx]
    test_mae = df["test_mae"].iloc[idx]; train_r_squared = df["train_r_squared"].iloc[idx]
    val_r_squared = df["val_r_squared"].iloc[idx]; test_r_squared = df["test_r_squared"].iloc[idx]
    results_dict = {"train_mae": [train_mae],
                    "val_mae": [val_mae],
                    "test_mae": [test_mae],
                    "train_r_squared": [train_r_squared],
                    "val_r_squared": [val_r_squared],
                    "test_r_squared": [test_r_squared]}

    return train_mae, val_mae, test_mae, train_r_squared, val_r_squared, test_r_squared


"""path = os.getcwd()
path_to_MLP_data = path + "/results/MLP/data/"

accumulated_relevances = [0.50 + i*0.05 for i in range(11)]
accumulated_relevances.append("full")

input_names = ["periodic", "nbo"]
target_names = ["barrier", "distance"]

for input_name in input_names:
    for target_name in target_names:
        summary_dict = {"train_mae": [],
                        "train_mae_stderr": [],
                        "val_mae": [],
                        "val_mae_stderr": [],
                        "test_mae": [],
                        "test_mae_stderr": [],
                        "train_r2": [],
                        "train_r2_stderr": [],
                        "val_r2": [],
                        "val_r2_stderr": [],
                        "test_r2": [],
                        "test_r2_stderr": [],
                        "best_mae_test": [],
                        "best_r2_test": [],
                        "nfeatures": [],
                        "accumulated_relevance": []}
        relevance = 0.1
        relevance_df = pd.read_csv(path_finder(input_name, target_name))
        for i, accumulated_relevance in enumerate(accumulated_relevances):
            tr_maes = []; v_maes = []; te_maes = []
            tr_maes_stderr = []; v_maes_stderr = []; te_maes_stderr = []
            tr_r2s = []; v_r2s = []; te_r2s = []
            tr_r2s_stderr = []; v_r2s_stderr = []; te_r2s_stderr = []
            for run in range(10):
                if accumulated_relevance == "full":
                    acc_rel = 1
                    relevance = -1
                    filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_full.csv"
                    if input_name == "periodic":
                        nfeatures = 671
                    else:
                        nfeatures = 2750
                elif accumulated_relevance > 0.992:
                    acc_rel = 1
                    relevance = 0
                    filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_100.csv"
                    nfeatures = np.sum(relevance_df["relevance"]>relevance)

                else:
                    acc_rel = 100*accumulated_relevance
                    relevance = find_accumulated_relevance(relevance_df, accumulated_relevance, relevance)
                    filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_{100*accumulated_relevance:.0f}.csv"
                    nfeatures = np.sum(relevance_df["relevance"]>relevance)
                df = pd.read_csv(path_to_gp_data + filename)
                train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
                tr_maes.append(train_mae); v_maes.append(val_mae); te_maes.append(test_mae)
                tr_r2s.append(train_r2); v_r2s.append(val_r2); te_r2s.append(test_r2)
            tr_mae = np.mean(tr_maes); v_mae = np.mean(v_maes); te_mae = np.mean(te_maes)
            tr_r2 = np.mean(tr_r2s); v_r2 = np.mean(v_r2s); te_r2 = np.mean(te_r2s)
            tr_mae_err = np.std(tr_maes)/np.sqrt(10); v_mae_err = np.std(v_maes)/np.sqrt(10)
            te_mae_err = np.std(te_maes)/np.sqrt(10); tr_r2_err = np.std(tr_r2s)/np.sqrt(10)
            v_r2_err = np.std(v_r2s)/np.sqrt(10); te_r2_err = np.std(te_r2s)/np.sqrt(10)
            best_idx = np.argmin(te_maes); best_mae = te_maes[best_idx]; best_r2 = te_r2s[best_idx]
            print(te_r2s)
            summary_dict["train_mae"].append(tr_mae); summary_dict["val_mae"].append(v_mae)
            summary_dict["test_mae"].append(te_mae); summary_dict["train_r2"].append(tr_r2)
            summary_dict["val_r2"].append(v_r2); summary_dict["test_r2"].append(te_r2)
            summary_dict["train_mae_stderr"].append(tr_mae_err); summary_dict["val_mae_stderr"].append(v_mae_err)
            summary_dict["test_mae_stderr"].append(te_mae_err); summary_dict["train_r2_stderr"].append(tr_r2_err)
            summary_dict["val_r2_stderr"].append(v_r2_err); summary_dict["test_r2_stderr"].append(te_r2_err)
            summary_dict["best_mae_test"].append(best_mae); summary_dict["best_r2_test"].append(best_r2)
            summary_dict["nfeatures"].append(nfeatures); summary_dict["accumulated_relevance"].append(acc_rel)

        df = pd.DataFrame(data=summary_dict)
        df.to_csv(path_to_MLP_data + "summary_" + input_name + "_" + target_name + ".csv")"""



def count_AC_type(df, accumulated_relevance):
    aa = 0
    bb = 0
    ab = 0
    relevance = find_accumulated_relevance(df, accumulated_relevance, 0.1)
    nfeatures = np.sum(df["relevance"]>relevance)
    for i in range(int(nfeatures)):
        if "AA" in df["feature"][i]:
            aa += 1
        elif "AB" in df["feature"][i]:
            ab += 1
        else:
            bb += 1

    return aa, bb, ab

"""accumulated_relevance = 0.80
target = "barrier"; input = "periodic"
df = pd.read_csv(path_finder(input, target))
aa, bb, ab = count_AC_type(df, accumulated_relevance)
print(input + " " + target + ":")
print(f"nAA: {aa}, nBB = {bb}, nAB = {ab}")
accumulated_relevance = 0.86
target = "barrier"; input = "nbo"
df = pd.read_csv(path_finder(input, target))
aa, bb, ab = count_AC_type(df, accumulated_relevance)
print(input + " " + target + ":")
print(f"nAA: {aa}, nBB = {bb}, nAB = {ab}")
accumulated_relevance = 0.82
target = "distance"; input = "periodic"
df = pd.read_csv(path_finder(input, target))
aa, bb, ab = count_AC_type(df, accumulated_relevance)
print(input + " " + target + ":")
print(f"nAA: {aa}, nBB = {bb}, nAB = {ab}")
accumulated_relevance = 0.52
target = "distance"; input = "nbo"
df = pd.read_csv(path_finder(input, target))
aa, bb, ab = count_AC_type(df, accumulated_relevance)
print(input + " " + target + ":")
print(f"nAA: {aa}, nBB = {bb}, nAB = {ab}")"""
"""path = os.getcwd()
path_to_MLP_data = path + "/results/MLP/data/"


input_names = ["periodic", "nbo"]
target_names = ["barrier", "distance"]

for input_name in input_names:
    for target_name in target_names:
        summary_dict = {"AA": [],
                        "BB": [],
                        "BA": [],
                        "train_mae": [],
                        "train_mae_stderr": [],
                        "val_mae": [],
                        "val_mae_stderr": [],
                        "test_mae": [],
                        "test_mae_stderr": [],
                        "train_r2": [],
                        "train_r2_stderr": [],
                        "val_r2": [],
                        "val_r2_stderr": [],
                        "test_r2": [],
                        "test_r2_stderr": [],
                        "best_mae_test": [],
                        "best_r2_test": [],
                        "nfeatures": [],
                        "accumulated_relevance": []}
        relevance = 0.1
        relevance_df = pd.read_csv(path_finder(input_name, target_name))
        if input_name == "periodic":
            if target_name == "barrier":
                accumulated_relevance = 0.80
            else:
                accumulated_relevance = 0.82
        else:
            if target_name == "barrier":
                accumulated_relevance = 0.86
            else:
                accumulated_relevance = 0.52

        aa, bb, ba = count_AC_type(relevance_df, accumulated_relevance)
        tr_maes = []; v_maes = []; te_maes = []
        tr_maes_stderr = []; v_maes_stderr = []; te_maes_stderr = []
        tr_r2s = []; v_r2s = []; te_r2s = []
        tr_r2s_stderr = []; v_r2s_stderr = []; te_r2s_stderr = []
        for run in range(10):
            if accumulated_relevance == "full":
                acc_rel = 1
                relevance = -1
                filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_full.csv"
                if input_name == "periodic":
                    nfeatures = 671
                else:
                    nfeatures = 2750
            elif accumulated_relevance > 0.992:
                acc_rel = 1
                relevance = 0
                filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_100.csv"
                nfeatures = np.sum(relevance_df["relevance"]>relevance)

            else:
                acc_rel = 100*accumulated_relevance
                relevance = find_accumulated_relevance(relevance_df, accumulated_relevance, relevance)
                filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_{100*accumulated_relevance:.0f}.csv"
                nfeatures = np.sum(relevance_df["relevance"]>relevance)
            df = pd.read_csv(path_to_MLP_data + filename)
            train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
            tr_maes.append(train_mae); v_maes.append(val_mae); te_maes.append(test_mae)
            tr_r2s.append(train_r2); v_r2s.append(val_r2); te_r2s.append(test_r2)
        tr_mae = np.mean(tr_maes); v_mae = np.mean(v_maes); te_mae = np.mean(te_maes)
        tr_r2 = np.mean(tr_r2s); v_r2 = np.mean(v_r2s); te_r2 = np.mean(te_r2s)
        tr_mae_err = np.std(tr_maes)/np.sqrt(10); v_mae_err = np.std(v_maes)/np.sqrt(10)
        te_mae_err = np.std(te_maes)/np.sqrt(10); tr_r2_err = np.std(tr_r2s)/np.sqrt(10)
        v_r2_err = np.std(v_r2s)/np.sqrt(10); te_r2_err = np.std(te_r2s)/np.sqrt(10)
        best_idx = np.argmin(te_maes); best_mae = te_maes[best_idx]; best_r2 = te_r2s[best_idx]
        print(te_r2s)
        summary_dict["AA"].append(aa); summary_dict["BB"].append(bb); summary_dict["BA"].append(ba)
        summary_dict["train_mae"].append(tr_mae); summary_dict["val_mae"].append(v_mae)
        summary_dict["test_mae"].append(te_mae); summary_dict["train_r2"].append(tr_r2)
        summary_dict["val_r2"].append(v_r2); summary_dict["test_r2"].append(te_r2)
        summary_dict["train_mae_stderr"].append(tr_mae_err); summary_dict["val_mae_stderr"].append(v_mae_err)
        summary_dict["test_mae_stderr"].append(te_mae_err); summary_dict["train_r2_stderr"].append(tr_r2_err)
        summary_dict["val_r2_stderr"].append(v_r2_err); summary_dict["test_r2_stderr"].append(te_r2_err)
        summary_dict["best_mae_test"].append(best_mae); summary_dict["best_r2_test"].append(best_r2)
        summary_dict["nfeatures"].append(nfeatures); summary_dict["accumulated_relevance"].append(acc_rel)

        df = pd.DataFrame(data=summary_dict)
        df.to_csv(path_to_MLP_data + "summary_optimal_" + input_name + "_" + target_name + ".csv")"""


path = os.getcwd()
path_to_MLP_data = path + "/results/MLP/data/"


input_names = ["periodic", "nbo"]
target_names = ["barrier", "distance"]

for input_name in input_names:
    for target_name in target_names:
        summary_dict = {"AA": [],
                        "BB": [],
                        "BA": [],
                        "train_mae": [],
                        "train_mae_stderr": [],
                        "val_mae": [],
                        "val_mae_stderr": [],
                        "test_mae": [],
                        "test_mae_stderr": [],
                        "train_r2": [],
                        "train_r2_stderr": [],
                        "val_r2": [],
                        "val_r2_stderr": [],
                        "test_r2": [],
                        "test_r2_stderr": [],
                        "best_mae_test": [],
                        "best_r2_test": [],
                        "nfeatures": [],
                        "accumulated_relevance": []}
        relevance = 0.1
        relevance_df = pd.read_csv(path_finder(input_name, target_name))
        if input_name == "periodic":
            if target_name == "barrier":
                accumulated_relevance = 0.80
            else:
                accumulated_relevance = 0.82
        else:
            if target_name == "barrier":
                accumulated_relevance = 0.86
            else:
                accumulated_relevance = 0.52

        aa, bb, ba = count_AC_type(relevance_df, accumulated_relevance)
        tr_maes = []; v_maes = []; te_maes = []
        tr_maes_stderr = []; v_maes_stderr = []; te_maes_stderr = []
        tr_r2s = []; v_r2s = []; te_r2s = []
        tr_r2s_stderr = []; v_r2s_stderr = []; te_r2s_stderr = []
        for run in range(10):
            filename = "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_optimal_trsz_20.csv"
            relevance = find_accumulated_relevance(relevance_df, accumulated_relevance, relevance)
            nfeatures = np.sum(relevance_df["relevance"] > relevance)
            df = pd.read_csv(path_to_MLP_data + filename)
            train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
            tr_maes.append(train_mae); v_maes.append(val_mae); te_maes.append(test_mae)
            tr_r2s.append(train_r2); v_r2s.append(val_r2); te_r2s.append(test_r2)
        tr_mae = np.mean(tr_maes); v_mae = np.mean(v_maes); te_mae = np.mean(te_maes)
        tr_r2 = np.mean(tr_r2s); v_r2 = np.mean(v_r2s); te_r2 = np.mean(te_r2s)
        tr_mae_err = np.std(tr_maes)/np.sqrt(10); v_mae_err = np.std(v_maes)/np.sqrt(10)
        te_mae_err = np.std(te_maes)/np.sqrt(10); tr_r2_err = np.std(tr_r2s)/np.sqrt(10)
        v_r2_err = np.std(v_r2s)/np.sqrt(10); te_r2_err = np.std(te_r2s)/np.sqrt(10)
        best_idx = np.argmin(te_maes); best_mae = te_maes[best_idx]; best_r2 = te_r2s[best_idx]
        print(te_r2s)
        print(te_maes)
        summary_dict["AA"].append(aa); summary_dict["BB"].append(bb); summary_dict["BA"].append(ba)
        summary_dict["train_mae"].append(tr_mae); summary_dict["val_mae"].append(v_mae)
        summary_dict["test_mae"].append(te_mae); summary_dict["train_r2"].append(tr_r2)
        summary_dict["val_r2"].append(v_r2); summary_dict["test_r2"].append(te_r2)
        summary_dict["train_mae_stderr"].append(tr_mae_err); summary_dict["val_mae_stderr"].append(v_mae_err)
        summary_dict["test_mae_stderr"].append(te_mae_err); summary_dict["train_r2_stderr"].append(tr_r2_err)
        summary_dict["val_r2_stderr"].append(v_r2_err); summary_dict["test_r2_stderr"].append(te_r2_err)
        summary_dict["best_mae_test"].append(best_mae); summary_dict["best_r2_test"].append(best_r2)
        summary_dict["nfeatures"].append(nfeatures); summary_dict["accumulated_relevance"].append(accumulated_relevance)

        df = pd.DataFrame(data=summary_dict)
        df.to_csv(path_to_MLP_data + "summary_optimal_trsz_20_" + input_name + "_" + target_name + ".csv")

for input_name in input_names:
    for target_name in target_names:
        df = pd.read_csv(path_to_MLP_data + "summary_optimal_trsz_20_" + input_name + "_" + target_name + ".csv")
        print(input_name + " " + target_name)
        columns = ["AA", "BB", "BA", "test_mae", "test_mae_stderr"]
        aa = df["AA"].values[0]; bb = df["BB"].values[0]; ba = df["BA"].values[0]
        mae = df["test_mae"].values[0]; mae_stderr = df["test_mae_stderr"].values[0]
        r2 = df["test_r2"].values[0]; r2_stderr = df["test_r2_stderr"].values[0]
        best_mae = df["best_mae_test"].values[0]; best_r2 = df["best_r2_test"].values[0]
        print(f"AA: {aa:.0f}, BB: {bb:.0f}, BA: {ba:.0f}")
        print(f"MAE = {mae} +- {1.96*mae_stderr}")
        print(f"R2 = {r2} +- {1.96*r2_stderr}")
        print(f"Best: MAE = {best_mae}, R2 = {best_r2}")
