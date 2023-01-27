""""Script for apply autocorrelation function"""

from multiprocessing import Pool, cpu_count
import time
import os
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import index

import csv

from graph_info import node_info, metal_index, edge_info, vector_feature_PT, vector_feature_NBO
from ac_funtions import *
from utilities import save_vectors, join_vectors, round_csv

# parameters
ac_operator = 'MR'
model_number = 1 # nBB model (1, 2, 3)
depth_max = 8
walk = 'AB' # AA, BBavg, BB, AB

def read_graph(file):

    # parameters (copy the same as above)
    ac_operator = 'MR'
    model_number = 1 # nBB model (1, 2, 3)
    depth_max = 8
    walk = 'AB' # AA, BBavg, BB, AB

    # computation dict
    comp_dict = {'MA': np.multiply,
                    'FA': np.multiply,
                    'MD': np.subtract,
                    'FD': np.subtract,
                    'MR': np.divide,
                    'FR': np.divide,
                    'MS': np.add,
                    'FS': np.add}

    # path to the gml file graphs
    #path = Path.home()/'Desktop/phd_stay_Project/uNatQ_graphs'
    path = Path.home()/'Desktop/phd_stay_Project/Vaskas_project/Vaskas_uNatQ_graphs/uNatQ_graphs'
    
    # feature NBO list
    feature_set_NBO = vector_feature_NBO(depth_max, ac_operator, walk)

    # unpack the feature labels
    feature_node_uNat, feature_edge_uNat, feature_edge_dNat, feature_node_uNat_depth,\
    feature_edge_uNat_depth, feature_node_dNat_depth, feature_edge_dNat_depth = feature_set_NBO

    # define the class graph
    G = nx.Graph()

    file = os.path.join(path, file)

    # read the graph
    G = nx.read_gml(file)

    # add feature_identity attribute to nodes and edges
    nx.set_node_attributes(G, 1, "feature_identity")
    nx.set_edge_attributes(G, 1, "feature_identity")

    # draw graphs
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()

    # set the starting node
    indx = metal_index(G)

    # walk over the attributes
    node_dict = node_info(G, depth_max, indx)
    edge_dict = edge_info(G, depth_max, indx)

    # perform AC function
    #AA_AC_vector, AA_AC_vname = atom_atom_MC(G, indx, depth_max, node_dict, feature_node_uNat, ac_operator, comp_dict)
    #AA_FA_vector, AA_FA_vname = atom_atom_F(G, depth_max, feature_node_uNat, ac_operator, comp_dict)
    
    #BB_AC_vector, BB_AC_vname, BB_AC_vector_avg, BB_AC_vname_avg  = bond_bond_MC(G, depth_max, edge_dict, feature_edge_uNat, ac_operator, comp_dict)
    #BB_FA_vector, BB_FA_vname = bond_bond_F(G, depth_max, feature_edge_uNat, ac_operator, comp_dict)
    
    AB_AC_vector, AB_AC_vname = bond_atom_MC(G, indx, depth_max, edge_dict, feature_node_uNat, feature_edge_uNat, ac_operator, comp_dict)
    #AB_FA_vector, AB_FA_vname = bond_atom_F(G, indx, depth_max, edge_dict, feature_node_uNat, feature_edge_uNat, ac_operator, comp_dict)

    return AB_AC_vname

if __name__ == "__main__":

    # time of execution
    start_time = time.time()
    
    # path to documents
    general_path = Path.home()/'Desktop/phd_stay_Project/coding'
    
    #path_to_gml = general_path/'../uNatQ_graphs'
    path_to_gml = general_path/'../Vaskas_project/Vaskas_UNatQ_graphs/uNatQ_graphs'
    #path_to_folder = general_path/f'../NBO_{walk}'
    path_to_folder = general_path/f'../Vaskas_project/NBO_{walk}'
    path_to_disconnected = general_path/'../excluded_graphs/disconnected.txt'
    path_to_test = general_path/'../test

    # exclude disconnected graphs
    disconnected_graphs = []
    with open(path_to_disconnected) as f:
        contents = f.readlines()
        for i in contents:
            disconnected_graphs.append(i.strip('\n'))

    # store gml_files
    gml_list = []
    for file in os.listdir(path_to_gml):
        if file.endswith('.gml'):
            if file not in disconnected_graphs:
                gml_list.append((path_to_gml/f'{file}'))
                #print(type(file), type(Path(file)))
                gml_list.append(f'{file}')

    # work with multiprocessing
    with Pool(processes=12) as pool:
        poolReturn = pool.map(read_graph, gml_list)

    feature_set_NBO = vector_feature_NBO(depth_max, ac_operator, walk)

    # unpack the feature labels
    feature_node_uNat, feature_edge_uNat, feature_edge_dNat, \
    feature_node_uNat_depth, feature_edge_uNat_depth, \
    feature_node_dNat_depth, feature_edge_dNat_depth = feature_set_NBO

    # select the feature type
    feature_type = feature_node_uNat_depth

    # join the vectors
    out_dict = join_vectors(poolReturn, feature_type)

    # save derived vectors in a .csv file
    print('Save vectors in a .csv file')
    save_vectors(path_to_folder, out_dict, depth_max, ac_operator, walk)

    # round vector values of the csv
    round_csv(path_to_folder, depth_max, ac_operator, walk)

    print("Execution time: " + str(round((time.time() - start_time)/60, 4)) + \
        " minutes." + str(cpu_count()))