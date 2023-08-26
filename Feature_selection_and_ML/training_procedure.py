from training_procedures import MLPTrainer
import torch
import pandas as pd
from models import DNN
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
from torch.optim.lr_scheduler import StepLR
from data_preprocessing import scale_features, CustomDataset, load_specific_split
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def set_global_seed(seed: int) -> None:

    """Sets the random seed for python, numpy and pytorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed = 2022
# set seed
set_global_seed(seed)

"""
Define paths
"""
path = os.getcwd()
path_to_relevance = path + "/results/reduced_autocorrelation_vectors/"
path_to_target = path + "/target_data/Vaska_vectors.csv"
periodic_relevance_barrier = "periodic_relevance_barrier.csv"
periodic_relevance_distance = "periodic_relevance_distance.csv"
nbo_relevance_barrier = "nbo_relevance_barrier.csv"
nbo_relevance_distance = "nbo_relevance_distance.csv"
data_path = path + "/data/autocorrelation_vectors/"
periodic = "AABBA_periodic_d6.csv"
nbo = "AABBA_NBO_d6.csv"
saving_path = path + "/data_to_perform_ML_on/"
data_saving_path = path + "/testing_MLP_results/data/"
"""
Lower bound on the relevance. Change to check multiple different ones.
"""
#relevance = -1 # by setting it to a negative number all AC descriptors are included
#relevance = 0 # only AC vectors with a relevance as calculated from the GBM
relevance = 0.00415552056576304 # Top 41 features for distance prediction using the periodic ACs

xtrain, xval, xtest, ytrain, yval, ytest, nfeatures, accumulated_relevance = load_specific_split(data_path + periodic,
                                                                                                 path_to_target,
                                                                                                 "target_distance",
                                                                                                 path_to_relevance + periodic_relevance_distance,
                                                                                                 relevance = relevance)


"""
Here I only scaled features, but I tried scaling the outputs aswell. However, this may be part of the problem.
"""
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


model = DNN(input_nodes=n_features, hidden_nodes=128, output_nodes=1)  #+len(feature_node_NBO)
# set up optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


trainer = MLPTrainer(model, optimizer, scheduler)
print('Starting training..')
n_epochs = 200
trained_model = trainer.run(train_loader,
                            train_loader_unshuffled,
                            val_loader, test_loader,
                            n_epochs=n_epochs)

df = trainer._training_information
#df.to_csv(data_saving_path + "training_information.csv")

plt.plot(df["epoch"], df["val_error"], label="Val")
plt.plot(df["epoch"], df["train_error"], label="Train")
plt.plot(df["epoch"], df["test_error"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("MAE [Å]")
plt.legend()
plt.savefig(path + "/testing_MLP_results/error_plot.pdf", format="pdf", bbox_inches="tight")
