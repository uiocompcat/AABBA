import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
from data_preprocessing import scale_features


def run_GBM(xtrain, xtest, ytrain, ytest, params):
    """Runs gradient boosting regression on defined training and test sets.
    Arguments:
        xtrain,     numpy.array(shape = (num_training_points, num_features))
            Training set features.
        xtest,      numpy.array(shape = (num_testing_points, num_features))
            Testing set features.
        ytrain,     numpy.array(shape = (num_training_points, ))
            Training set targets.
        ytest,      numpy.array(shape = (num_testing_points, ))
            Testing set targets.
        params,         dict
            Dictionary containing all parameters wanted specified for the
            GradientBoostingRegressor model given by scikit-learn.
    """

    reg = GradientBoostingRegressor(**params)
    reg.fit(xtrain, ytrain)
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    mae_score_test = np.zeros((params["n_estimators"],), dtype=np.float64)
    mae_score_train = np.zeros((params["n_estimators"],), dtype=np.float64)
    r2_train = np.zeros((params["n_estimators"],), dtype=np.float64)
    r2_test = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, ypred in enumerate(reg.staged_predict(xtest)):
        test_score[i] = mean_squared_error(ytest, ypred)
        r2_test[i] = r2_score(ytest, ypred)
        mae_score_test[i] = np.mean(abs(ytest-ypred))

    for i, ypred in enumerate(reg.staged_predict(xtrain)):
        r2_train[i] = r2_score(ytrain, ypred)
        mae_score_train[i] = np.mean(abs(ytrain-ypred))

    best_idx = np.argmin(mae_score_test)
    ypred = reg.predict(xtest)
    data_best = {"true": ytest,
                 "pred": ypred}
    return reg.feature_importances_, reg.train_score_, test_score, mae_score_train, mae_score_test, r2_train, r2_test, data_best


def make_folds(k, df, target, seed=2023):
    """Cross-validation of GBM model.
    Arguments:
        k,               int
            The number of folds.
        df,              pd.DataFrame
            Dataset.
        target,          string
            Column name of target.
    Returns:
        folds,          list of lists
            Lists containing the indices of each fold.
    """

    n, p = df.shape
    rng = default_rng(seed)
    indices = rng.choice(n, n, replace=False)
    k_size = int(n/k)
    remainder = n%k
    # Adds one index to the remainder first folds
    k_sizes = [k_size + 1 for i in range(remainder)]
    for i in range(k-remainder):
        k_sizes.append(k_size)
    folds = []
    counter = 0
    for i, ksize in enumerate(k_sizes):
        fold=[]
        for j in range(ksize):
            fold.append(indices[counter])
            counter += 1
        folds.append(fold)
    return folds

def cross_validation_GBM(k, df, target, params, seed=2023):
    """
    Arguments:
        k,              int
            Number of folds in the cross-validation.
        df,             pd.DataFrame
            Dataset.
        target,         string
            Feature name that is target in dataset.
        params,         dict
            Dictionary containing all parameters wanted specified for the
            GradientBoostingRegressor model given by scikit-learn.
        seed,           int
            Integer specifying seed for fold-making.
    Returns:
        feature_importances,
            data with feature importance (relevance).
        train_scores,           numpy.ndarray(shape = (k, boosting_iterations))
            Training MSE scores from all runs for all boosting iterations.
        test_scores,            numpy.ndarray(shape = (k, boosting_iterations))
            Testing MSE scores from all runs for all boosting iterations.
        mae_scores_train,           numpy.ndarray(shape = (k, boosting_iterations))
            Training MAE scores from all runs for all boosting iterations.
        mae_scores,            numpy.ndarray(shape = (k, boosting_iterations))
            Testing MAE scores from all runs for all boosting iterations.
        r2_scores_train,           numpy.ndarray(shape = (k, boosting_iterations))
            Training R2 scores from all runs for all boosting iterations.
        r2_scores,            numpy.ndarray(shape = (k, boosting_iterations))
            Testing R2 scores from all runs for all boosting iterations.
        best,                 dict
            Dictionary containing the predictions and targets for the best models (full models unless overfit). 
    """
    folds = make_folds(k, df, target, seed=seed)
    feature_importances = []
    train_scores = []
    test_scores = []
    mae_scores = []
    mae_scores_train = []
    r2_scores = []
    r2_scores_train = []
    data_best = []
    for fold_nr, fold in enumerate(folds):
        fold_nrs = [i for i in range(k) if i != fold_nr]
        test_idxs = folds[fold_nr]
        test = df.loc[test_idxs]
        train = pd.concat([df.loc[folds[fold]] for fold in fold_nrs], axis=0)
        xtest = test.drop(columns="target")
        xtrain = train.drop(columns="target")
        ytest = test[target]
        ytrain = train[target]
        xtrain, xtest, xscaler = scale_features(xtrain, xtest)
        FI, train_score, test_score, mae_score_train, mae_score_test, r2_tr, r2_te, best = run_GBM(xtrain, xtest, ytrain, ytest, params)
        feature_importances.append(FI); train_scores.append(train_score)
        test_scores.append(test_score); mae_scores.append(mae_score_test)
        mae_scores_train.append(mae_score_train); r2_scores_train.append(r2_tr)
        r2_scores.append(r2_te); data_best.append(best)

    return feature_importances, train_scores, test_scores, mae_scores, mae_scores_train, r2_scores_train, r2_scores, data_best

if __name__ == "__main__":
    df = np.zeros((12, 10))
    target = "string"
    k = 5
    folds = make_folds(k, df, target)
    print(folds)
