import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



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
    removals = ["id"]
    df = df.drop(columns=removals)
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
