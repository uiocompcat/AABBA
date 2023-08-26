import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
"""
Lucía Moran and Hannes Kneiding's CustomDataset class.
"""

class CustomDataset(Dataset):

    def __init__(self, x, y):

        """Constructor for a custom dataset. This object will serve as the data provider for the torch functions.

        Args:
            x (tensor): A torch tensor of the features values.
            y (tensor): A torch tensor of the target values.
        """

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



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

def load_specific_split(path_to_ac, path_to_target, target, path_to_relevance, relevance=1e-5):
    """
    Loads the data split in split_data/
    """

    df_target = pd.read_csv(path_to_target)
    df = pd.read_csv(path_to_ac)
    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])

    df = df.dropna(axis=1)
    df["target"] = target_vector
    """
    Loading specific split
    """
    path_to_split = "/home/jeb/Desktop/AABBA_Paper/data_preprocessing/data_split/"
    training_names = pd.read_csv(path_to_split + "file_name_train.csv")
    test_names = pd.read_csv(path_to_split + "file_name_test.csv")
    val_names = pd.read_csv(path_to_split + "file_name_val.csv")
    xtrain = df[df["id"].isin(training_names["names"])]
    xval = df[df["id"].isin(val_names["names"])]
    xtest = df[df["id"].isin(test_names["names"])]
    ytrain = xtrain["target"]; yval = xval["target"]; ytest = xtest["target"]
    if relevance < 0:
        """
        If the relevance is a negative number then we only drop the redundant features.
        """
        xtrain = xtrain.drop(columns=["target", "id", "Unnamed: 0"])
        xtest = xtest.drop(columns=["target", "id", "Unnamed: 0"])
        xval = xval.drop(columns=["target", "id", "Unnamed: 0"])
        features = xtrain.columns
        for feature in features:
            if xtrain[feature].std() == 0 and feature != "id":
                xtrain = xtrain.drop(columns=feature)
                xtest = xtest.drop(columns=feature)
                xval = xval.drop(columns=feature)
    else:
        """
        Gather features by relevance.
        """
        relevance_df = pd.read_csv(path_to_relevance)
        xtrain = xtrain[relevance_df["feature"][relevance_df["relevance"]>relevance]]
        xtest = xtest[relevance_df["feature"][relevance_df["relevance"]>relevance]]
        xval = xval[relevance_df["feature"][relevance_df["relevance"]>relevance]]
    nfeatures = xtrain.shape[1]
    if relevance < 0:
        accumulated_relevance = 1
    else:
        accumulated_relevance = np.sum(relevance_df["relevance"][relevance_df["relevance"]>relevance])

    return xtrain, xval, xtest, ytrain, yval, ytest, nfeatures, accumulated_relevance

def transform_data(xtrain, ytrain, xtest, ytest, n_components=None, use_pca=False):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardized
    and low-dimensional train and test sets together with the scaler object for the
    target values.
    Arguments:
        xtrain: size=(ntrain, p),
            training input
        ytrain: size=(ntrain, ?),
            training truth, ? depends on what we train against (mulitple objective)
        xtest: size=(ntest, p),
            testing input
        ytest: size=(ntest, ?),
            testing truth
        n_components: int,
            number of principal components used if use_pca=True
        use_pca: bool,
            if true use principal component analysis for dimensionality reduction

    Returns:
        xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, yscaler
    """

    xscaler = StandardScaler()
    xtrain_scaled = xscaler.fit_transform(xtrain)
    xtest_scaled = xscaler.transform(xtest)
    yscaler = StandardScaler()
    ytrain_scaled = yscaler.fit_transform(ytrain)
    ytest_scaled = yscaler.transform(ytest)

    if use_pca:
        pca = PCA(n_components)
        xtrain_scaled = pca.fit_transform(xtrain)
        print("Fraction of variance retained is: ", np.sum(pca.explained_variance_ratio_))
        xtest_scaled = pca.transform(xtest)
    return xtrain_scaled, xtest_scaled, ytrain_scaled, ytest_scaled, yscaler

def load_data(data_path, target_path, target):
    df_target = pd.read_csv(target_path)
    df = pd.read_csv(data_path)
    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])
    #removals = ["id"]
    #df = df.drop(columns=removals)
    df = df.dropna(axis=1)
    return df, target_vector

def load_data_cv(data_path, target_path, target):
    df_target = pd.read_csv(target_path)
    df = pd.read_csv(data_path)
    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])
    removals = ["id"]
    df = df.drop(columns=removals)
    df = df.dropna(axis=1)
    df["target"] = target_vector
    return df

def scale_features(xtrain, xtest):
    xscaler = StandardScaler()
    xtrain_scaled = xscaler.fit_transform(xtrain)
    xtest_scaled = xscaler.transform(xtest)
    return xtrain_scaled, xtest_scaled, xscaler

def load_to_loaders(path_to_ac, path_to_target, target, path_to_relevance, relevance=1e-5, batch_size=32):
    xtrain, xval, xtest, ytrain, yval, ytest, nfeatures = load_specific_split(path_to_ac,
                                                                               path_to_target,
                                                                               target,
                                                                               path_to_relevance=path_to_relevance,
                                                                               relevance = relevance)

    """
    Standardize features
    """
    xtrain, xval, xscaler = scale_features(xtrain, xval)
    xtest = xscaler.transform(xtest)

    xtrain_torch = torch.tensor(xtrain, dtype=torch.float64)
    ytrain_torch = torch.tensor(ytrain.values.reshape((-1, 1)), dtype=torch.float64)
    dataset_train = CustomDataset(xtrain_torch, ytrain_torch)

    xval_torch = torch.tensor(xval, dtype=torch.float64)
    yval_torch = torch.tensor(yval.values.reshape((-1, 1)), dtype=torch.float64)
    dataset_val = CustomDataset(xval_torch, yval_torch)

    xtest_torch = torch.tensor(xtest, dtype=torch.float64)
    ytest_torch = torch.tensor(ytest.values.reshape((-1, 1)), dtype=torch.float64)
    dataset_test = CustomDataset(xtest_torch, ytest_torch)


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_unshuffled = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, train_loader_unshuffled, val_loader, test_loader, nfeatures


if __name__ == "__main__":
    """
    Loading figure and data path, and colors used in plotting.
    """
    fig_path = "/home/jeb/Desktop/ABBA_Paper/results/MLP/figures/"
    data_path = "/home/jeb/Desktop/ABBA_Paper/data/autocorrelation_vectors"
    data_saving_path = "/home/jeb/Desktop/ABBA_Paper/results/MLP/data/"
    target_path = "/home/jeb/Desktop/ABBA_Paper/ac_generation/data_Vaska/data_27_april/"
    target_gp = "gpVaska_vectors.csv"
    target_nbo = "nboVaska_vectors.csv"
    path_saving_features = "/home/jeb/Desktop/ABBA_Paper/results/reduced_autocorrelation_vectors/"
    split_path = "/home/jeb/Desktop/ABBA_Paper/data_preprocessing/data_split/"
    train = "file_name_train.csv"; val = "file_name_val.csv"; test = "file_name_test.csv"
    gp = "/ABBA_GP_d6.csv"
    nbo = "/ABBA_NBO_d6.csv"
    path_to_ac = data_path + gp
    path_to_target = target_path + target_gp
    path_to_relevance = path_saving_features + "gp_relevance_barrier.csv"
    batch_size=32
    relevance = 1e-4
    train_loader, train_loader_unshuffled, val_loader, test_loader, nfeatures = load_to_loaders(path_to_ac,
                                                                                                path_to_target,
                                                                                                "target_barrier",
                                                                                                path_to_relevance,
                                                                                                batch_size = batch_size,
                                                                                                relevance = relevance)
    print("Train loader")
    for batch_idx, batch in enumerate(train_loader):
        print("Batch index: ", batch_idx)
        print("Batch shape: ", batch)

    print("Train loader unshuffled")
    for batch_idx, batch in enumerate(train_loader_unshuffled):
        print("Batch index: ", batch_idx)
        print("Batch shape: ", batch)
    print("Validation loader")
    for batch_idx, batch in enumerate(val_loader):
        print("Batch index: ", batch_idx)
        print("Batch shape: ", batch)
    print("Test loader")
    for batch_idx, batch in enumerate(test_loader):
        print("Batch index: ", batch_idx)
        print("Batch shape: ", batch)
