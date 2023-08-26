"""
Imports
"""
import pandas as pd
import numpy as np
import os
from training_procedures import BaseTrainer, MLPTrainer
import matplotlib.pyplot as plt
import tqdm
import torch
import pandas as pd
from models import ExampleNet
import matplotlib.pyplot as plt
from data_preprocessing import scale_features, CustomDataset, load_specific_split
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


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

def find_accumulated_relevance(df, acc_rel_lb, start_relevance):
    """Finds the relevance bound given an accumulated relevance lower bound.
    Arguments
        df,                 pd.DataFrame
            DataFrame containing relevances of features.
        acc_rel_lb,         float
            Lower bound for accumulated relevance.
        start_relevance,    float
            Start point for finding the accumulated relevance.

    Returns
        relevance,          float
            Relevance value corresponding to the lower bound
            for the accumulated relevance.
    """
    if start_relevance > 1e-3:
        eps = 1e-4
    else:
        eps = 1e-6
    relevance = start_relevance - eps
    counter = 0
    while np.sum(df["relevance"][df["relevance"]>relevance])< acc_rel_lb:
        relevance -= eps
        counter += 1

    return relevance

def run(input_name, target_name, accumulated_relevance):
    # Where to store data
    data_saving_path = os.getcwd() + "/results/MLP/data/"
    autocorrelation_vectors_path = os.getcwd() + "/data/autocorrelation_vectors/"
    path_to_target = os.getcwd() + "/target_data/Vaska_vectors.csv"
    n_neurons = 128
    n_layers = 2

    if accumulated_relevance == 1:
        relevance = 0
        relevance_tag = "100"
    elif accumulated_relevance == "full":
        relevance = -1
        relevance_tag = "full"
    else:
        relevance = find_accumulated_relevance(pd.read_csv(path_finder(input_name, target_name)), accumulated_relevance, 0.1)
        relevance_tag = f"{100*accumulated_relevance:.0f}"
    if input_name == "periodic":
        ACs = "AABBA_periodic_d6.csv"
    else:
        ACs = "AABBA_NBO_d6.csv"


    xtrain, xval, xtest, ytrain, yval, ytest, nfeatures, accumulated_relevance = load_specific_split(autocorrelation_vectors_path + ACs,
                                                                                                     path_to_target,
                                                                                                     "target_" + target_name,
                                                                                                     path_finder(input_name, target_name),
                                                                                                     relevance = relevance)


    xtrain, xtest, xscaler = scale_features(xtrain, xtest)
    xval = xscaler.transform(xval)
    xtrain = torch.tensor(xtrain, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values.reshape((-1, 1)), dtype=torch.float)
    xval = torch.tensor(xval, dtype=torch.float)
    yval = torch.tensor(yval.values.reshape((-1, 1)), dtype=torch.float)
    xtest = torch.tensor(xtest, dtype=torch.float)
    ytest = torch.tensor(ytest.values.reshape((-1, 1)), dtype=torch.float)
    n_features = xtrain.shape[-1]
    batch_size = 32
    dataset_train = CustomDataset(xtrain, ytrain)
    dataset_val = CustomDataset(xval, yval)
    dataset_test = CustomDataset(xtest, ytest)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_unshuffled = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    learning_rate = 0.01
    runs = 10
    for run in range(runs):
        model = ExampleNet(input_nodes = n_features, hidden_nodes = n_neurons, output_nodes = 1, hidden_layers = n_layers)
        # set up optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


        trainer = MLPTrainer(model, optimizer, scheduler)
        print(f'Run {run + 1}')
        n_epochs = 200
        trained_model = trainer.run(train_loader,
                                    train_loader_unshuffled,
                                    val_loader, test_loader,
                                    n_epochs=n_epochs)

        df = trainer._training_information
        df.to_csv(data_saving_path + "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_relevance_" + relevance_tag + ".csv")
        idx_min = np.argmin(df["val_error"])
        val_error = df["val_error"].iloc[idx_min]; train_error = df["train_error"].iloc[idx_min]
        test_error = df["test_error"].iloc[idx_min]; r2_val = df["val_r_squared"].iloc[idx_min]
        r2_train = df["train_r_squared"].iloc[idx_min]; r2_test = df["test_r_squared"].iloc[idx_min]
        print("MAE")
        print(f"Train = {train_error:.4f}, Val = {val_error:.4f}, test = {test_error:.4f}")
        print("R2")
        print(f"Train = {r2_train:.4f}, Val = {r2_val:.4f}, test = {r2_test:.4f}")
    return 0


def run_optimal_reverse(input_name, target_name, accumulated_relevance):
    """Trains and saves performance to file for a dense neural network using optimal input.
    20:40:40 split.
    Arguments
        input_name,                 string
            'periodic' or 'nbo'
        target_name,                string
            'barrier' or 'distance'
        accumulated_relevance,      float
            Lower bound for the accumulated relevance of the input. As
            calculated by the GBM.
    """

    path = os.getcwd()
    autocorrelation_vectors_path = path + "/data/autocorrelation_vectors/"
    path_to_target = path + "/target_data/Vaska_vectors.csv"
    data_saving_path = path + "/results/MLP/data/"
    relevance = find_accumulated_relevance(pd.read_csv(path_finder(input_name, target_name)), accumulated_relevance, 0.1)
    relevance_name = "optimal"
    if input_name == "periodic":
        ACs = "AABBA_periodic_d6.csv"
    else:
        ACs = "AABBA_NBO_d6.csv"


    xtest, xtrain_1, xtrain_2, ytest, ytrain_1, ytrain_2, nfeatures, accumulated_relevance = load_specific_split(autocorrelation_vectors_path + ACs,
                                                                                                                 path_to_target,
                                                                                                                 "target_" + target_name,
                                                                                                                 path_finder(input_name, target_name),
                                                                                                                 relevance = relevance)
    x_test = xtest.iloc[0:int(xtest.shape[0]/2), :]; y_test = ytest.iloc[0:int(xtest.shape[0]/2)]
    x_val = xtest.iloc[int(xtest.shape[0]/2):-1, :]; y_val = ytest.iloc[int(xtest.shape[0]/2):-1]
    x_train = pd.concat([xtrain_1, xtrain_2])
    y_train = pd.concat([ytrain_1, ytrain_2])
    xtrain, xtest, xscaler = scale_features(x_train, x_test)
    xval = xscaler.transform(x_val)
    xtrain = torch.tensor(xtrain, dtype=torch.float)
    ytrain = torch.tensor(y_train.values.reshape((-1, 1)), dtype=torch.float)
    xval = torch.tensor(xval, dtype=torch.float)
    yval = torch.tensor(y_val.values.reshape((-1, 1)), dtype=torch.float)
    xtest = torch.tensor(xtest, dtype=torch.float)
    ytest = torch.tensor(y_test.values.reshape((-1, 1)), dtype=torch.float)
    n_features = xtrain.shape[-1]
    n_neurons = 128
    n_layers = 2
    batch_size = 32
    dataset_train = CustomDataset(xtrain, ytrain)
    dataset_val = CustomDataset(xval, yval)
    dataset_test = CustomDataset(xtest, ytest)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_unshuffled = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    learning_rate = 0.01
    runs = 10
    for run in range(runs):
        model = ExampleNet(input_nodes = n_features, hidden_nodes = n_neurons, output_nodes = 1, hidden_layers = n_layers)
        # set up optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


        trainer = MLPTrainer(model, optimizer, scheduler)
        print(f'Run {run + 1}')
        n_epochs = 200
        trained_model = trainer.run(train_loader,
                                    train_loader_unshuffled,
                                    val_loader, test_loader,
                                    n_epochs=n_epochs)

        df = trainer._training_information
        df.to_csv(data_saving_path + "MLP_" + input_name + "_" + target_name + f"_run_{run+1:.0f}_optimal_trsz_20.csv")
        idx_min = np.argmin(df["val_error"])
        val_error = df["val_error"].iloc[idx_min]; train_error = df["train_error"].iloc[idx_min]
        test_error = df["test_error"].iloc[idx_min]; r2_val = df["val_r_squared"].iloc[idx_min]
        r2_train = df["train_r_squared"].iloc[idx_min]; r2_test = df["test_r_squared"].iloc[idx_min]
        print("MAE")
        print(f"Train = {train_error:.4f}, Val = {val_error:.4f}, test = {test_error:.4f}")
        print("R2")
        print(f"Train = {r2_train:.4f}, Val = {r2_val:.4f}, test = {r2_test:.4f}")

        fig = plt.figure()
        plt.plot(df["epoch"], df["val_error"], label="Val")
        plt.plot(df["epoch"], df["train_error"], label="Train")
        plt.plot(df["epoch"], df["test_error"], label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("MAE [Å]")
        plt.legend()
        plt.savefig(os.getcwd() + "/results/MLP/figures/bayes_opt_MLP/" + input_name + "_" + target_name + f"_run_{run:.0f}.pdf", format="pdf", bbox_inches="tight")





"""
Bayes optimized runs on optimal input.
"""

"""
run_optimal_bayes_opt("periodic", "barrier")
run_optimal_bayes_opt("periodic", "distance")
run_optimal_bayes_opt("nbo", "barrier")
run_optimal_bayes_opt("nbo", "distance")
"""

"""
Runs
"""
"""
accumulated_relevances = [0.50, 0.55, 0.65, 0.75]
inputs = ["periodic", "nbo"]
targets = ["barrier", "distance"]
for input_name in inputs:
    for target_name in targets:
        for accumulated_relevance in accumulated_relevances:
            run(input_name, target_name, accumulated_relevance)

"""
"""
Specific runs (best from gaussian process runs)
"""
#accumulated_relevance = 0.80
#target = "barrier"; input = "periodic"
#run(input, target, accumulated_relevance)
#run_optimal_reverse(input, target, accumulated_relevance)
accumulated_relevance = 0.86
target = "barrier"; input = "nbo"
#run(input, target, accumulated_relevance)
run_optimal_reverse(input, target, accumulated_relevance)
#accumulated_relevance = 0.82
#target = "distance"; input = "periodic"
#run(input, target, accumulated_relevance)
#run_optimal_reverse(input, target, accumulated_relevance)
#accumulated_relevance = 0.52
#target = "distance"; input = "nbo"
#run(input, target, accumulated_relevance)
#run_optimal_reverse(input, target, accumulated_relevance)
