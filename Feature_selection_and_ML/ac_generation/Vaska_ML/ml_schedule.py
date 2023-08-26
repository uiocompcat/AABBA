import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd

import tools
from nets import ExampleNet
from trainer import Trainer
from dataset import CustomDataset
from plot import plot_correlation, plot_target_histogram

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import processing_data
from pathlib import Path
import os

import csv
import schedule
import time

# PARAMETERS
wandb_entity = 'lmoran'

seed = 2022
learning_rate = 0.01
batch_size = 32
n_epochs = 200

#val_set_size = 0.1
#test_set_size = 0.1

def run_job(depth_max, ac_type_1, ac_type_2, ac_type_3, ac_type_4, model_number_1, walk_1, walk_2, walk_3, target, property, dnn, idx_train, idx_val, idx_test):

    print('new iteratio of the running')

    wandb_project_name = target
    wandb_run_name = f'{ac_type_1}_{walk_1}{walk_2}{walk_3}_d{depth_max}_{property}_e{n_epochs}_b{batch_size}_dnn{dnn}_2HL_seed2022' #'FA_AA_d8_homo_lumo_gap_delta_epoch200_b32'

    depth_max = depth_max
    ac_type_1 = ac_type_1
    ac_type_2 = ac_type_2
    ac_type_3 = ac_type_3
    ac_type_4 = ac_type_4
    model_number_1 = model_number_1
    walk_1 = walk_1
    walk_2 = walk_2
    walk_3 = walk_3

    # CODE
    wandb.init(project=wandb_project_name, entity=wandb_entity)
    # set name
    wandb.run.name = wandb_run_name

    # set seed
    tools.set_global_seed(seed)

    # setup data set
    data = pd.read_csv('../data_Vaska/gpVaska_vectors.csv') #gp data            './ac_vectors_target.csv')
    #dataNBO = pd.read_csv('../nboVaska_vectors.csv') #nbo data                 './ac_vectors_target.csv')

    # input generator
    feature_set_PT_1 = processing_data.vector_feature_PT(depth_max, ac_type_1, model_number_1, walk_1)
    feature_set_PT_2 = processing_data.vector_feature_PT(depth_max, ac_type_1, model_number_1, walk_2)
    feature_set_PT_3 = processing_data.vector_feature_PT(depth_max, ac_type_1, model_number_1, walk_3)
    feature_set_PT_4 = processing_data.vector_feature_PT(depth_max, ac_type_1, model_number_1, walk_1)

    feature_set_NBO_1 = processing_data.vector_feature_NBO(depth_max, ac_type_1, walk_1)
    feature_set_NBO_2 = processing_data.vector_feature_NBO(depth_max, ac_type_1, walk_2)
    feature_set_NBO_3 = processing_data.vector_feature_NBO(depth_max, ac_type_1, walk_3)
    feature_set_NBO_4 = processing_data.vector_feature_NBO(depth_max, ac_type_4, walk_1)

    # unpack the feature labels GP/PT
    feature_node_1, feature_edge_1, feature_node_depth_1, feature_edge_depth_1, \
    feature_new1_edge_depth_1, feature_new2_edge_depth_1, feature_new3_edge_depth_1 = feature_set_PT_1

    feature_node_2, feature_edge_2, feature_node_depth_2, feature_edge_depth_2, \
    feature_new1_edge_depth_2, feature_new2_edge_depth_2, feature_new3_edge_depth_2 = feature_set_PT_2

    feature_node_3, feature_edge_3, feature_node_depth_3, feature_edge_depth_3, \
    feature_new1_edge_depth_3, feature_new2_edge_depth_3, feature_new3_edge_depth_3 = feature_set_PT_3

    feature_node, feature_edge, feature_node_depth_4, feature_edge_depth_4, \
    feature_new1_edge_depth, feature_new2_edge_depth, feature_new3_edge_depth = feature_set_PT_4

    # unpack the feature labels NBO
    feature_node_uNat_1, feature_edge_uNat_1, feature_edge_dNat_1, \
    feature_node_uNat_depth_1, feature_edge_uNat_depth_1, \
    feature_node_dNat_depth_1, feature_edge_dNat_depth_1 = feature_set_NBO_1

    feature_node_uNat_2, feature_edge_uNat_2, feature_edge_dNat_2, \
    feature_node_uNat_depth_2, feature_edge_uNat_depth_2, \
    feature_node_dNat_depth_2, feature_edge_dNat_depth_2 = feature_set_NBO_2

    feature_node_uNat_3, feature_edge_uNat_3, feature_edge_dNat_3,\
    feature_node_uNat_depth_3, feature_edge_uNat_depth_3,\
    feature_node_dNat_depth_3, feature_edge_dNat_depth_3 = feature_set_NBO_3

    feature_node_uNat_4, feature_edge_uNat_4, feature_edge_dNat_4, feature_node_uNat_depth_4, \
    feature_edge_uNat_depth_4, feature_node_dNat_depth_1, feature_edge_dNat_depth_1 = feature_set_NBO_4

    # to remove and add specific features (ABBA_D)
    CR_MA_AA =  [f'S-{i}_{ac_type_1}_{walk_1}' for i in range(depth_max + 1)]
    CR_MR_AA =  [f'S-{i}_{ac_type_4}_{walk_1}' for i in range(depth_max + 1)]
    CR_MA_AB =  [f'S-{i}_{ac_type_1}_{walk_3}' for i in range(depth_max + 1)]
    CR_MR_AB =  [f'S-{i}_{ac_type_4}_{walk_3}' for i in range(depth_max + 1)]

    # features to use
    #features_input = feature_node_uNat_depth_1 + feature_edge_uNat_depth_2 + feature_node_uNat_depth_3  #feature_node_depth_1 + feature_edge_depth_2 + feature_node_depth_3 + CR_MR_AA + CR_MR_AB 
    features_input = feature_node_depth_1 + feature_edge_depth_2 + feature_node_depth_3                  # + CR_MR_AA + CR_MR_AB # - CR_MA_AA - CR_MA_AB  
    #features_input = [i for i in features_input if i not in  CR_MA_AA and i  not in CR_MA_AB]

    # select the data
    sub = data[features_input]
    #sub = pd.concat([data[features_input],  dataNBO[feature_node_NBO]], axis=1)
    target = data[property]

    print('len input', len(features_input), features_input, len(sub), sub, len(target), target)

    # standard scale subselection
    #sub = (sub - sub.mean()) / sub.std()

    # separate data in train, val and test with sckit-learn
    X_train, y_train = sub.iloc[idx_train], target.iloc[idx_train]
    X_val, y_val = sub.iloc[idx_val], target.iloc[idx_val]
    X_test, y_test = sub.iloc[idx_test], target.iloc[idx_test]

    print('train', X_train, y_train)
    print('val', X_val, y_val)
    print('test', X_test, y_test)

    #X_train, X_test, y_train, y_test = train_test_split(sub, target, test_size=0.2, train_size=0.8, random_state=2022, shuffle=True)
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2022, shuffle=True)

    # save train, val and set data, path to save the documents
    #general_path_csv = f'/mnt/c/Users/lucia/Desktop/phd_stay_Project/Vaskas_project/data/runs/{wandb_run_name}' 
    #os.mkdir(general_path_csv)
#    
    # standard scale subselection
    X_train, mean, std, outliers = processing_data.standarize_train(X_train)
    X_val = processing_data.standarize_rest(X_val, mean, std, outliers)
    X_test = processing_data.standarize_rest(X_test, mean, std, outliers)

    X_train_csv = X_train.to_csv(f'xtrain_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')
    y_train_csv = y_train.to_csv(f'ytrain_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')
    X_val_csv = X_val.to_csv(f'xval_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')
    y_val_csv = y_val.to_csv(f'yval_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')
    X_test_csv = X_test.to_csv(f'xtest_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')
    y_test_csv = y_test.to_csv(f'ytest_{ac_type_1}_{walk_1}_d{depth_max}_{wandb_project_name}.csv')

    print('subset', len(sub), 'train', len(X_train), X_train, 'val', len(X_val), X_val, 'test', len(X_test), X_test)

    # cast variables to torch tensors
    X_train_torch = torch.tensor(X_train.values, dtype=torch.float)
    y_train_torch = torch.tensor(y_train.values.reshape((-1, 1)), dtype=torch.float)
    dataset_train = CustomDataset(X_train_torch, y_train_torch)

    X_val_torch = torch.tensor(X_val.values, dtype=torch.float)
    y_val_torch = torch.tensor(y_val.values.reshape((-1, 1)), dtype=torch.float)
    dataset_val = CustomDataset(X_val_torch, y_val_torch)

    X_test_torch = torch.tensor(X_test.values, dtype=torch.float)
    y_test_torch = torch.tensor(y_test.values.reshape((-1, 1)), dtype=torch.float)
    dataset_test = CustomDataset(X_test_torch, y_test_torch)

    # divide into subsets for tmQMg
    #x_tensor = torch.tensor(sub.values, dtype=torch.float)
    #y_tensor = torch.tensor(target.values.reshape((-1, 1)), dtype=torch.float)
    #dataset = CustomDataset(x_tensor, y_tensor)

    # divide into subsets
    #sets = torch.utils.data.random_split(dataset, [len(dataset) - round(val_set_size * len(dataset)) - round(test_set_size * len(dataset)), round(val_set_size * len(dataset)), round(test_set_size * len(dataset))])

    #print('Using ' + str(len(dataset)) + ' data points. (train=' + str(len(sets[0])) + ', val=' + str(len(sets[1])) + ', test=' + str(len(sets[2])) + ')')
    print('Using ' + str(len(sub)) + ' data points. (train=' + str(len(X_train_torch)) + ', val=' + str(len(X_val_torch)) + ', test=' + str(len(X_test_torch)) + ')')

    # set up dataloaders for tmQMg
    #train_loader = DataLoader(sets[0], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #train_loader_unshuffled = DataLoader(sets[0], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #val_loader = DataLoader(sets[1], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #test_loader = DataLoader(sets[2], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_unshuffled = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if dnn == 1:
        hidden_nodes = 128
    if dnn == 2:
        hidden_nodes = 256

    # set up model
    model = ExampleNet(input_nodes=len(features_input)-len(outliers), hidden_nodes=hidden_nodes, output_nodes=1)  #+len(feature_node_NBO)
    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

    # run
    trainer = Trainer(model, optimizer, scheduler)
    print('Starting training..')

    trained_model = trainer.run(train_loader,
                                train_loader_unshuffled,
                                val_loader, test_loader,
                                n_epochs=n_epochs)

    # get training set predictions and ground truths
    train_predicted_values = trainer.predict_loader(train_loader_unshuffled)
    train_true_values = tools.get_target_list(train_loader_unshuffled)

    # get validation set predictions and ground truths
    val_predicted_values = trainer.predict_loader(val_loader)
    val_true_values = tools.get_target_list(val_loader)

    # get test set predictions and ground truths
    test_predicted_values = trainer.predict_loader(test_loader)
    test_true_values = tools.get_target_list(test_loader)

    # log predictions

    train_df = pd.DataFrame({'predicted': train_predicted_values,
    'truth': train_true_values})
    wandb.log({"train-predictions": wandb.Table(dataframe=train_df)})

    val_df = pd.DataFrame({'predicted': val_predicted_values,
    'truth': val_true_values})
    wandb.log({"val-predictions": wandb.Table(dataframe=val_df)})

    test_df = pd.DataFrame({'predicted': test_predicted_values,
    'truth': test_true_values})
    wandb.log({"test-predictions": wandb.Table(dataframe=test_df)})

    # log plots

    tmp_file_path = '/tmp/image.png'

    plot_correlation(train_predicted_values, train_true_values, file_path=tmp_file_path)
    wandb.log({'Training set prediction correlation': wandb.Image(tmp_file_path)})

    plot_correlation(val_predicted_values, val_true_values, file_path=tmp_file_path)
    wandb.log({'Validation set prediction correlation': wandb.Image(tmp_file_path)})

    plot_correlation(test_predicted_values, test_true_values, file_path=tmp_file_path)
    wandb.log({'Test set prediction correlation': wandb.Image(tmp_file_path)})

    plot_target_histogram(train_true_values, val_true_values, test_true_values, file_path=tmp_file_path)
    wandb.log({'Target value distributions': wandb.Image(tmp_file_path)})

    # end run
    wandb.finish(exit_code=0)


# Add the delay_seconds argument to run the jobs with a number
# of seconds delay in between.

if __name__ == "__main__":

    general_path = '/mnt/c/Users/lucia/Desktop/phd_stay_Project/Vaskas_project/'

    path_to_all = general_path + 'Vaskas_uNatQ_graphs/uNatQ_graphs'
    path_to_train = general_path + 'Vaskas_uNatQ_graphs_split/train/uNatQ_graphs'
    path_to_val = general_path + 'Vaskas_uNatQ_graphs_split/val/uNatQ_graphs'
    path_to_test = general_path + 'Vaskas_uNatQ_graphs_split/test/uNatQ_graphs'

    #path_to_all = general_path + 'baseline_graphs'
    #path_to_train = general_path + 'baseline_graphs_split/train'
    #path_to_val = general_path + 'baseline_graphs_split/val'
    #path_to_test = general_path + 'baseline_graphs_split/test'

    file_names = [f'{file[:-4]}' for file in os.listdir(path_to_all) if file.endswith('.gml')]
    file_name_train = [f'{file[:-4]}' for file in os.listdir(path_to_train) if file.endswith('.gml')]
    file_name_val = [f'{file[:-4]}' for file in os.listdir(path_to_val) if file.endswith('.gml')]
    file_name_test = [f'{file[:-4]}' for file in os.listdir(path_to_test) if file.endswith('.gml')]

    idx_train = [idx for idx, item in enumerate(file_names) if item in file_name_train]
    idx_val = [idx for idx, item in enumerate(file_names) if item in file_name_val]
    idx_test = [idx for idx, item in enumerate(file_names) if item in file_name_test]

    # run jobs sequentially
    schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='target_barrier_seed2022', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    #schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='gp_target_barrier_ABBA_D1', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    #schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='gp_target_barrier_ABBA_D1', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    #schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=3, walk_1='AA', walk_2='BBavg', walk_3='AB', target='gp_target_barrier_ABBA_D1', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    #schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=2, walk_1='BBavg2', walk_2='BB', walk_3='AB', target='gp_target_barrier_BBavg2', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    #schedule.every(20).minutes.do(run_job, depth_max=3, ac_type_1='MA', ac_type_2='MD', ac_type_3='MS', ac_type_4='MR', model_number_1=2, walk_1='BBavg2', walk_2='BB', walk_3='AB', target='gp_target_barrier_BBavg2', property='target_barrier', dnn=1, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    schedule.run_all()
