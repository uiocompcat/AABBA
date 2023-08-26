"""
Imports
"""
import pandas as pd
import numpy as np
import os
from data_preprocessing import load_specific_split, scale_features
from training_procedures import GPTrainer
from models import GP
import matplotlib.pyplot as plt
import tqdm
import torch
import gpytorch
import warnings

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
    """Trains and saves performance to file for a Gaussian process.
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
    data_saving_path = path + "/results/GP/data/"
    if accumulated_relevance == 1:
        relevance = 0
        relevance_name = "100_percent"
    elif accumulated_relevance == "full":
        relevance = -1
        relevance_name = "full"
    else:
        relevance = find_accumulated_relevance(pd.read_csv(path_finder(input_name, target_name)), accumulated_relevance, 0.1)
        relevance_name = f"{100*accumulated_relevance:.0f}_percent"
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
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    xval = torch.tensor(xval, dtype=torch.float)
    yval = torch.tensor(yval.values, dtype=torch.float)
    xtest = torch.tensor(xtest, dtype=torch.float)
    ytest = torch.tensor(ytest.values, dtype=torch.float)
    n_features = xtrain.shape[-1]
    # Likelihood
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(2e-4*torch.ones(xtrain.shape[0]),
                                                                   learn_additional_noise=True)
    # Model
    lr = 1.0
    n_epochs = 50
    model = GP(xtrain, ytrain, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.001)
    trainer = GPTrainer(model, optimizer, likelihood, scheduler = scheduler)
    model = trainer.run(xtrain, ytrain, xval, yval, xtest, ytest, n_epochs = n_epochs)
    df = trainer.training_info()
    df.to_csv(data_saving_path + "GP_" + input_name + "_" + target_name + "_relevance_" + relevance_name + ".csv")
    return model

def run_optimal_reverse(input_name, target_name, accumulated_relevance):
    """Trains and saves performance to file for a Gaussian process using optimal input.
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
    data_saving_path = path + "/results/GP/data/"
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
    x_val = xtest.iloc[0:int(xtest.shape[0]/2), :]; y_val = ytest.iloc[0:int(xtest.shape[0]/2)]
    x_test = xtest.iloc[int(xtest.shape[0]/2):-1, :]; y_test = ytest.iloc[int(xtest.shape[0]/2):-1]
    x_train = pd.concat([xtrain_1, xtrain_2])
    ytrain = pd.concat([ytrain_1, ytrain_2])
    xtrain, xtest, xscaler = scale_features(x_train, x_test)
    xval = xscaler.transform(x_val)
    xtrain = torch.tensor(xtrain, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    xval = torch.tensor(xval, dtype=torch.float)
    yval = torch.tensor(y_val.values, dtype=torch.float)
    xtest = torch.tensor(xtest, dtype=torch.float)
    ytest = torch.tensor(y_test.values, dtype=torch.float)
    n_features = xtrain.shape[-1]
    # Likelihood
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(2e-4*torch.ones(xtrain.shape[0]),
                                                                   learn_additional_noise=True)
    # Model
    lr = 1.0
    n_epochs = 100
    model = GP(xtrain, ytrain, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.001)
    trainer = GPTrainer(model, optimizer, likelihood, scheduler = scheduler)
    model = trainer.run(xtrain, ytrain, xval, yval, xtest, ytest, n_epochs = n_epochs)
    df = trainer.training_info()
    df.to_csv(data_saving_path + "optimal_runs/GP_" + input_name + "_" + target_name + "_relevance_" + relevance_name + "_trsz_20.csv")
    return model

def run_optimal(input_name, target_name, accumulated_relevance):
    """Trains and saves performance to file for a Gaussian process using optimal input.
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
    data_saving_path = path + "/results/GP/data/"
    relevance = find_accumulated_relevance(pd.read_csv(path_finder(input_name, target_name)), accumulated_relevance, 0.1)
    relevance_name = "optimal"
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
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    xval = torch.tensor(xval, dtype=torch.float)
    yval = torch.tensor(yval.values, dtype=torch.float)
    xtest = torch.tensor(xtest, dtype=torch.float)
    ytest = torch.tensor(ytest.values, dtype=torch.float)
    n_features = xtrain.shape[-1]
    # Likelihood
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(2e-4*torch.ones(xtrain.shape[0]),
                                                                   learn_additional_noise=True)
    # Model
    lr = 1.0
    n_epochs = 100
    model = GP(xtrain, ytrain, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.001)
    trainer = GPTrainer(model, optimizer, likelihood, scheduler = scheduler)
    model = trainer.run(xtrain, ytrain, xval, yval, xtest, ytest, n_epochs = n_epochs)
    df = trainer.training_info()
    df.to_csv(data_saving_path + "optimal_runs/GP_" + input_name + "_" + target_name + "_relevance_" + relevance_name + ".csv")
    return model





"""
Runs to probe optimal relevance
"""
"""
inputs = ["nbo", "periodic"]
targets = ["barrier"]
# edit to appropriate range of accumulated relevances
accumulated_relevances = [0.50 + i*0.01 for i in range(50)]
for input_name in inputs:
    for target_name in targets:
        for accumulated_relevance in accumulated_relevances:
            run(input_name, target_name, accumulated_relevance)
"""

"""
Runs optimal
"""

run_optimal("periodic", "barrier", 0.80)
run_optimal("periodic", "distance", 0.82)
run_optimal("nbo", "barrier", 0.86)
run_optimal("nbo", "distance", 0.52)

"""
Runs optimal 20:40:40
"""

run_optimal_reverse("periodic", "barrier", 0.80)
run_optimal_reverse("periodic", "distance", 0.82)
run_optimal_reverse("nbo", "barrier", 0.86)
run_optimal_reverse("nbo", "distance", 0.52)
