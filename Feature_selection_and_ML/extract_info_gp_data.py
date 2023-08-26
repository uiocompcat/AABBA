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
path_to_gp_data = path + "/results/GP/data/optimal_runs/"

accumulated_relevances = [0.50 + i*0.01 for i in range(51)]
accumulated_relevances.append("full")

input_names = ["periodic", "nbo"]
target_names = ["barrier", "distance"]

for input_name in input_names:
    for target_name in target_names:
        summary_dict = {"train_mae": [],
                        "val_mae": [],
                        "test_mae": [],
                        "train_r2": [],
                        "val_r2": [],
                        "test_r2": [],
                        "nfeatures": [],
                        "accumulated_relevance": []}
        relevance = 0.1
        relevance_df = pd.read_csv(path_finder(input_name, target_name))
        for i, accumulated_relevance in enumerate(accumulated_relevances):
            if accumulated_relevance == "full":
                acc_rel = 1
                relevance = -1
                filename = "GP_" + input_name + "_" + target_name + "_relevance_full.csv"
                if input_name == "periodic":
                    nfeatures = 671
                else:
                    nfeatures = 2750
            elif accumulated_relevance > 0.992:
                acc_rel = 1
                relevance = 0
                filename = "GP_" + input_name + "_" + target_name + "_relevance_100_percent.csv"
                nfeatures = np.sum(relevance_df["relevance"]>relevance)

            else:
                acc_rel = 100*accumulated_relevance
                relevance = find_accumulated_relevance(relevance_df, accumulated_relevance, relevance)
                filename = "GP_" + input_name + "_" + target_name + f"_relevance_{100*accumulated_relevance:.0f}_percent.csv"
                nfeatures = np.sum(relevance_df["relevance"]>relevance)
            df = pd.read_csv(path_to_gp_data + filename)
            train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
            summary_dict["train_mae"].append(train_mae); summary_dict["val_mae"].append(val_mae)
            summary_dict["test_mae"].append(test_mae); summary_dict["train_r2"].append(train_r2)
            summary_dict["val_r2"].append(val_r2); summary_dict["test_r2"].append(test_r2)
            summary_dict["nfeatures"].append(nfeatures); summary_dict["accumulated_relevance"].append(acc_rel)

        df = pd.DataFrame(data=summary_dict)
        df.to_csv(path_to_gp_data + "summary_" + input_name + "_" + target_name + ".csv")"""

path = os.getcwd()
path_to_optimal_data = path + "/results/GP/data/optimal_runs/"

input_names = ["periodic", "nbo"]
target_names = ["barrier", "distance"]
for input_name in input_names:
    for target_name in target_names:
        filename = "GP_" + input_name + "_" + target_name + "_relevance_optimal.csv"
        summary_dict = {"train_mae": [],
                        "val_mae": [],
                        "test_mae": [],
                        "train_r2": [],
                        "val_r2": [],
                        "test_r2": []
                        }
        df = pd.read_csv(path_to_optimal_data + filename)
        train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
        summary_dict["train_mae"].append(train_mae); summary_dict["val_mae"].append(val_mae)
        summary_dict["test_mae"].append(test_mae); summary_dict["train_r2"].append(train_r2)
        summary_dict["val_r2"].append(val_r2); summary_dict["test_r2"].append(test_r2)

        print(input_name + " " + target_name)
        print(f"MAE = {test_mae}, R2 = {test_r2}")

for input_name in input_names:
    for target_name in target_names:
        filename = "GP_" + input_name + "_" + target_name + "_relevance_optimal_trsz_20.csv"
        summary_dict = {"train_mae": [],
                        "val_mae": [],
                        "test_mae": [],
                        "train_r2": [],
                        "val_r2": [],
                        "test_r2": []
                        }
        df = pd.read_csv(path_to_optimal_data + filename)
        train_mae, val_mae, test_mae, train_r2, val_r2, test_r2 = extract_scores(df)
        summary_dict["train_mae"].append(train_mae); summary_dict["val_mae"].append(val_mae)
        summary_dict["test_mae"].append(test_mae); summary_dict["train_r2"].append(train_r2)
        summary_dict["val_r2"].append(val_r2); summary_dict["test_r2"].append(test_r2)

        print(input_name + " " + target_name)
        print(f"MAE = {test_mae}, R2 = {test_r2}")
