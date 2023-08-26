from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from data_preprocessing import transform_data, load_data, scale_features, load_data_cv
from training_procedures import make_folds, run_GBM, cross_validation_GBM
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import tqdm

"""
The plots will need to be modified with respect to target for ylabel. 
"""

def set_font_size_figures(fontsize):
    params = {'legend.fontsize': fontsize,
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize': fontsize,
         'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

set_font_size_figures("medium")

"""
Loading figure and data path.
"""
path_to_here = os.getcwd()
fig_path = path_to_here + "/results/feature_importance/figures/"
data_path = path_to_here + "/data/autocorrelation_vectors/"
data_saving_path = path_to_here + "/results/feature_importance/data/"
target_path = path_to_here + "/target_data/Vaska_vectors.csv"
path_saving_features = path_to_here + "/results/reduced_autocorrelation_vectors/"
"""
Names of autocorrelation vectors
"""
# AABBA vectors of depth 6
periodic = "/ABBA_GP_d6.csv"
nbo = "/ABBA_NBO_d6.csv"
# atom-atom vectors of depth 6
aa_periodic = "/AA_GP_d6.csv"
aa_nbo = "/AA_NBO_d6.csv"
# bond-bond vectors of depth 6
bb_periodic = "/BB_GP_d6.csv"
bb_nbo = "/BB_NBO_d6.csv"
# bond-atom vectors of depth 6
ab_periodic = "/AB_GP_d6.csv"
ab_nbo = "/AB_NBO_d6.csv"

n_estimators = 1000
target = "target_barrier" # Starting with barrier
"""
Change periodic to whatever you want to run here, and target can be changed to
'target_distance' if you want to test on the prediction of the H---H distances
instead.
"""
df = load_data_cv(data_path + periodic, target_path, target)
df = df.drop(columns="Unnamed: 0")

params = {
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.05,
        "loss": "squared_error",
        }

# Number of folds in cross-validation
k = 5

"""
k-fold cross-validation
"""
fis, mses_tr, mses, maes, maes_tr, r2s_tr, r2s_te, best_data = cross_validation_GBM(k, df, "target", params)
test_size = len(best_data[0]["true"])
preds_and_truths = {}
for run, best_data_run in enumerate(best_data):
    data_preds = best_data_run["pred"]
    data_truth = best_data_run["true"].to_numpy()
    #print(type(data_truth))
    if len(data_truth) != test_size:
        data_preds = np.concatenate((data_preds, np.array([10])))
        data_truth = np.concatenate((data_truth, np.array([10])))
    #print("preds length: ", len(data_preds))
    #print("truth length: ", len(data_truth))
    preds_and_truths[f"preds_{str(run+1)}"] = data_preds
    preds_and_truths[f"truths_{str(run+1)}"] = data_truth

"""
Save predictions and truths data
"""
#print(preds_and_truths)
df_preds_and_truths = pd.DataFrame(data=preds_and_truths)
df_preds_and_truths.to_csv(data_saving_path + "GP_predsVtruths_barrier.csv")

"""
Scatter Preds vs Truth
"""
for i in range(k):
    fig = plt.figure()
    plt.scatter(preds_and_truths[f"truths_{str(i+1)}"],preds_and_truths[f"preds_{str(i+1)}"], color="tab:blue", alpha=0.3)
    plt.xlabel("DFT calculated barrier [kcal/mol]")
    plt.ylabel("Predicted barrier [kcal/mol]")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(fig_path + f"GP_GBM_pairplot_barrier_{i:.0f}.pdf", format="pdf", bbox_inches="tight")

"""
Saving data about runs
"""

argmins = np.zeros(k)
for i in range(k):
    argmins[i] = int(np.argmin(maes[i]))
data_dict_info = {"run": [i+1 for i in range(k)],
                      "best_mae": [maes[i][int(argmins[i])] for i in range(k)],
                      "best_mae_train": [maes_tr[i][int(argmins[i])] for i in range(k)],
                      "best_mse": [mses[i][int(argmins[i])] for i in range(k)],
                      "best_mse_train": [mses_tr[i][int(argmins[i])] for i in range(k)],
                      "best_r2": [r2s_te[i][int(argmins[i])] for i in range(k)],
                      "best_r2_train": [r2s_tr[i][int(argmins[i])] for i in range(k)]
                     }

df_run_specifics = pd.DataFrame(data=data_dict_info)
df_run_specifics.to_csv(data_saving_path + "GP_barrier_run_specifications.csv")

data_dict_run_info = {"boosting_iteration": [i+1 for i in range(params["n_estimators"])]}

for i in range(k):
    data_dict_run_info[f"mae_run_{i+1}"] = maes[i]
    data_dict_run_info[f"mae_train_run_{i+1}"] = maes_tr[i]
    data_dict_run_info[f"r2_run_{i+1}"] = r2s_te[i]
    data_dict_run_info[f"r2_train_run_{i+1}"] = r2s_tr[i]
    data_dict_run_info[f"mse_run_{i+1}"] = mses[i]
    data_dict_run_info[f"mse_train_run_{i+1}"] = mses_tr[i]

df_run_data = pd.DataFrame(data=data_dict_run_info)
df_run_data.to_csv(data_saving_path + "GP_barrier_runs.csv")

"""
Calculating standard errors of the means
"""
r2_test_seotm = np.std(r2s_te, axis=0)/np.sqrt(k)
r2_train_seotm = np.std(r2s_tr, axis=0)/np.sqrt(k)
mae_test_seotm = np.std(maes, axis=0)/np.sqrt(k)
mae_train_seotm = np.std(maes_tr, axis=0)/np.sqrt(k)
mse_test_seotm = np.std(mses, axis=0)/np.sqrt(k)
mse_train_seotm = np.std(mses_tr, axis=0)/np.sqrt(k)

"""
Feature importances
"""
feature_importances = np.mean(fis, axis=0)
feature_importances_seotm = np.std(fis, axis=0)/np.sqrt(k)

"""
R2 plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(r2s_tr, axis=0),
         color="tab:blue",
         label="R2 training")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(r2s_tr, axis=0) - 1.96*r2_train_seotm,
                np.mean(r2s_tr, axis=0) + 1.96*r2_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(r2s_te, axis=0),
         color="tab:orange",
         label="R2 testing")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(r2s_te, axis=0) - 1.96*r2_test_seotm,
                np.mean(r2s_te, axis=0) + 1.96*r2_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="lower right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"R$^2$ barrier")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + "GP_R2_GBR_barrier.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
MSE plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(mses_tr, axis=0),
         color="tab:blue",
         label="MSE train")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(mses_tr, axis=0) - 1.96*mse_train_seotm,
                np.mean(mses_tr, axis=0) + 1.96*mse_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(mses_tr, axis=0),
         color="tab:orange",
         label="MSE test")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(mses, axis=0) - 1.96*mse_test_seotm,
                np.mean(mses, axis=0) + 1.96*mse_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="upper right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"MSE barrier [(kcal/mol)$^2$]")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + "GP_MSE_GBR_barrier.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
MAE plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(maes_tr, axis=0),
         color="tab:blue",
         label="MAE train")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(maes_tr, axis=0) - 1.96*mae_train_seotm,
                np.mean(maes_tr, axis=0) + 1.96*mae_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(maes, axis=0),
         color="tab:orange",
         label="MAE test")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(maes, axis=0) - 1.96*mae_test_seotm,
                np.mean(maes, axis=0) + 1.96*mae_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="upper right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"MAE barrier [kcal/mol]")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + "GP_MAE_GBR_barrier.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
Feature importance
"""
sorted_idx = feature_importances.argsort()
sorted_idx = np.flip(sorted_idx)
pos = np.arange(sorted_idx[0:20].shape[0]) + 0.5

relevance_dictionary = {"feature": df.columns[sorted_idx],
                        "relevance": feature_importances[sorted_idx],
                        "relevance_seotm": feature_importances_seotm[sorted_idx]}
relevance_data = pd.DataFrame(data=relevance_dictionary)
relevance_data.to_csv(path_or_buf=path_saving_features + "/gp_relevance_barrier.csv")

"""
Bar plot
"""
set_font_size_figures("medium")
fig = plt.figure()
plt.bar(pos, 100*feature_importances[sorted_idx[0:20]],
        align="center",
        color="tab:blue",
        yerr=100*feature_importances_seotm[sorted_idx[0:20]],
        linewidth=1,
        edgecolor="white",
        alpha=1.0)
plt.xticks(pos, np.array(df.columns[sorted_idx[0:20]]))
plt.ylabel("AC descriptor importance in GBM for barrier [%]", fontsize="medium")
plt.xlabel("AC descriptor", fontsize="medium")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.xticks(rotation="vertical")
plt.savefig(fig_path + "GP_feature_importance_barrier.pdf", format="pdf", bbox_inches="tight")
plt.show()
best_idx = np.argmin(np.mean(maes, axis=0))
print("-"*50)
print(f"Best MAE barrier periodic set: {np.mean(maes, axis=0)[best_idx]}+-{1.96*mae_test_seotm[best_idx]}")
print(f"Best MAE barrier training periodic set: {np.mean(maes_tr, axis=0)[best_idx]}+-{1.96*mae_train_seotm[best_idx]}")
print(f"Best R2 barrier periodic set: {np.mean(r2s_te, axis=0)[best_idx]}+-{1.96*r2_test_seotm[best_idx]}")
print(f"Best R2 barrier training periodic set: {np.mean(r2s_tr, axis=0)[best_idx]}+-{1.96*r2_train_seotm[best_idx]}")
print("-"*50)
