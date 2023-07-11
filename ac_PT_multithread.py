""""Script for apply autocorrelation function"""

from multiprocessing import Pool, cpu_count
import time
import os
from pathlib import Path

import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from graph_info import node_info, metal_index, edge_info, vector_feature_PT, vector_feature_NBO
from ac_funtions import *
from utilities import round_csv, save_vectors, join_vectors, round_numbers

# parameters
ac_operator = 'MA'
model_number = 3 # nBB model for PT (1, 2, 3)
depth_max = 8
walk = 'ABBAavg' # AA, BBavg, BB, AB, ABBA

def read_graph(file):

    # parameters (copy the same as above)
    ac_operator = 'MA'
    model_number = 3 # nBB model (1, 2, 3) for GP and (4, 5) for NBO properties
    depth_max = 8
    walk = 'ABBAavg' # AA, BBavg, BB, AB, ABBAavg, ABBA

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
    path = Path.home()/'Desktop/phd_stay_Project/Vaskas_project/baseline_graphs'

    # feature list
    feature_set_PT = vector_feature_PT(depth_max, ac_operator, model_number, walk)

    # unpack the feature labels
    feature_node, feature_edge, feature_node_depth, feature_edge_depth, \
    feature_new1_edge_depth, feature_new2_edge_depth, feature_new3_edge_depth = feature_set_PT

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
    #AA_AC_vector, AA_AC_vname = atom_atom_MC(G, indx, depth_max, node_dict, feature_node, ac_operator, comp_dict)
    #AA_FA_vector, AA_FA_vname = atom_atom_F(G, depth_max, feature_node, ac_operator, comp_dict)
    
    #BB_AC_vector, BB_AC_vname, BB_AC_vector_avg, BB_AC_vname_avg  = bond_bond_MC(G, depth_max, edge_dict, feature_edge, ac_operator, comp_dict)
    #BB_FA_vector, BB_FA_vname = bond_bond_F(G, depth_max, feature_edge, ac_operator, comp_dict)
    
    #AB_AC_vector, AB_AC_vname = bond_atom_MC(G, indx, depth_max, edge_dict, feature_node, feature_edge, ac_operator, comp_dict)
    #AB_FA_vector, AB_FA_vname = bond_atom_F(G, depth_max, feature_node, feature_edge, ac_operator, comp_dict)

    nBB_AC_vector, nBB_AC_vname = new_gp_bond_bond_MC(G, depth_max, node_dict, edge_dict, model_number, ac_operator, comp_dict)
    #nBB_FA_vector, nBB_FA_vname = new_gp_bond_bond_F(G, depth_max, node_dict, model_number, ac_operator, comp_dict)

    return nBB_AC_vname # return vector with the name of the graphs

if __name__ == "__main__":

    # time of execution
    start_time = time.time()

    # path to documents
    general_path = Path.home()/'Desktop/phd_stay_Project/'
    
    path_to_gml = general_path/'Vaskas_project/baseline_graphs'
    path_to_folder = general_path/f'Vaskas_project/PT_{walk}'
    path_to_disconnected = general_path/'excluded_graphs/disconnected.txt'
    path_to_test = general_path/'test'

    # exclude disconnected graphs
    #disconnected_graphs = []
    #with open(path_to_disconnected) as f:
    #    contents = f.readlines()
    #    for i in contents:
    #        disconnected_graphs.append(i.strip('\n'))

    # store gml_files
    gml_list = [] # list
    for file in os.listdir(path_to_gml):
        if file.endswith('.gml'):    ##('.gml'):
            #if file not in disconnected_graphs:    ##('.gml'):
                #gml_list.append((path_to_gml/f'{file}'))
                #print(type(file), type(Path(file)))
                gml_list.append(f'{file}')

    # save maximum depth
    #get_max_depth(gml_list)

    # create a process
    with Pool(processes=12-4) as pool:
        poolReturn = pool.map(read_graph, gml_list) #, depth_max, ac_operator, model_number, *feature_set)

    feature_set_PT = vector_feature_PT(depth_max, ac_operator, model_number, walk)

    # unpack the feature labels
    feature_node, feature_edge, feature_node_depth, feature_edge_depth, \
    feature_new1_edge_depth, feature_new2_edge_depth, feature_new3_edge_depth = feature_set_PT

    # select the feature type
    feature_type = feature_new3_edge_depth

    # save derived vectors in a .csv file
    print('Save vectors in a .csv file')

    # join the vectors and save vectors
    out_dict = join_vectors(poolReturn, feature_type)
    save_vectors(path_to_test, out_dict, depth_max, ac_operator, walk, model_number)

    # round features of the csv
    round_csv(path_to_test, depth_max, ac_operator, walk, model_number)

    print("Execution time: " + str(round((time.time() - start_time)/60, 4)) + \
        " minutes." + str(cpu_count()))