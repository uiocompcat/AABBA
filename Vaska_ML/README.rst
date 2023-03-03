Hei, this is the ML code I have been using. 

This is a Hannes' code and then I made some adaptations to fit our dataset. 
Hannes suggested that I use wandb (Weights and Biases), 
which has proven to be very useful for saving DNN results and tracking metrics.


For runnning the DNNs (tmQMg) I used the ml_schedule.py and dataset.py that are in the tmQMg directory.
For running the DNNs for the Vaska's complex the one that are on the Vaska's directory.

The main distinction is that while the split is carried out using PyTorch in the case of tmQMg, separation has already been established for Vaska's complex. 
Additionally, I can include a text file containing the names for the train, validation, and test sets for the Vaska's complex. 
It is worth noting that I maintained the same split throughout all of the experiments.


In the ml_schedule.py file, there are the guidelines for selecting a particular set of features as input. 
It may be cumbersome but it follows the same logic as the one used in the development of the autocorrelation functions. 

In the last lines of the script, you can define the input model. Example: 

schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='target_barrier_seed2022', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

depth_max: especify the maximum depth of the features
ac_type_1: especify the type of autocorrelation function (MA: metal-centered multiplied properties)
ac_type_2: especify the type of autocorrelation function (MD: metal-centered deltametric properties)
ac_type_3: especify the type of autocorrelation function (MS: metal-centered summatric properties)
ac_type_4: especify the type of autocorrelation function (MR: metal-centered random properties)
model_number_1: especify the number of the model (1, 2, 3) for the BB1avg, BB2avg, BB3avg features
walk_1: especify the type of walk (AA: atom-atom correlation)
walk_2: especify the type of walk (BBavg: average bond-bond correlation)
walk_3: especify the type of walk (AB: atom-bond correlation)
target: especify the target (target_barrier_seed2022: this is the label that I used to save the model in the project with this name in wandb)
property: especify the property (target_barrier: the name of the target feature in the dataset)
dnn: especify the nodes of DNN (1: 128 nodes, 2: 256 nodes)
idx_train: especify the train set (idx_train: the labels of the train set)
idx_val: especify the validation set (idx_val: the labels of the validation set)
idx_test: especify the test set (idx_test: the labels of the test set)

In the line 111, you have to specify the features. 

For example, if you want to the Atom x Atom properties and the Atom x Bond properties in the multiplied form, you have to especify the following:

schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='target_barrier_seed2022', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

feature_node_1, feature_edge_1, feature_node_depth_1, feature_edge_depth_1, \
feature_new1_edge_depth_1, feature_new2_edge_depth_1, feature_new3_edge_depth_1 = feature_set_PT_1
feature_node_3, feature_edge_3, feature_node_depth_3, feature_edge_depth_3, \
feature_new1_edge_depth_3, feature_new2_edge_depth_3, feature_new3_edge_depth_3 = feature_set_PT_3

features_input = feature_node_depth_1 + feature_node_depth_3

print(features_input) 
['Z-0_MA_AA', 'Z-1_MA_AA', 'Z-2_MA_AA', 'Z-3_MA_AA', 'I-0_MA_AA', 'I-1_MA_AA', 'I-2_MA_AA', 'I-3_MA_AA', 'T-0_MA_AA', 
'T-1_MA_AA', 'T-2_MA_AA', 'T-3_MA_AA', 'S-0_MA_AA', 'S-1_MA_AA', 'S-2_MA_AA', 'S-3_MA_AA', 
'chi-0_MA_AA', 'chi-1_MA_AA', 'chi-2_MA_AA', 'chi-3_MA_AA',
'Z-0_MA_AB', 'Z-1_MA_AB', 'Z-2_MA_AB', 'Z-3_MA_AB', 'I-0_MA_AB', 'I-1_MA_AB', 'I-2_MA_AB', 'I-3_MA_AB',
'T-0_MA_AB', 'T-1_MA_AB', 'T-2_MA_AB', 'T-3_MA_AB', 'S-0_MA_AB', 'S-1_MA_AB', 'S-2_MA_AB', 'S-3_MA_AB',
'chi-0_MA_AB', 'chi-1_MA_AB', 'chi-2_MA_AB', 'chi-3_MD_AB']

In the nets.py file, you can find the DNN architectures. There is a DNN with 3HL using ReLU. 
I usually comment two lines for running a DNN with 2HL. 

The trainer.py file contains code for training the DNNs, including the run function where the loss function is defined and the early stopping criteria for test error is set.

processingdata.py contains functions for performing the standarization of the data and the acquisition of the labels of the features (to match with the .csv)

I hope this helps.
