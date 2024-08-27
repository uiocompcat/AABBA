""""Script for apply autocorrelation function"""

from multiprocessing import Pool, cpu_count
import time
import os
from pathlib import Path

import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import index

from graph_info import node_info, metal_index, edge_info, nbo_new_edge_attribute, vector_feature_NBO
from ac_funtions import *
from utilities import save_vectors, join_vectors, round_csv

# paramteres to modify
PARAMS = {
    'ac_operator': 'MA',
    'model_number': 4,  # AABBA(II) model (4, 5) for NBO, any number for the AA, BBavg, BB, or AB
    'depth_max': 8,
    'walk': 'ABBAavg'  # AA, BBavg, BB, AB, ABBAavg, ABBA
}

# Define feature set mapping functions
def get_feature_for_walk(walk, fs, model_number):
    if walk == 'AA':
        return fs[3]  # feature_node_depth
    elif walk in ['BBavg', 'BB']:
        return fs[4]  # feature_edge_depth
    elif walk == 'AB':
        return fs[3]  # feature_edge_depth
    elif walk in ['ABBAavg', 'ABBA']:
        # Return the feature based on model_number
        return fs[5 + model_number]  # feature_new1_edge_depth, feature_new2_edge_depth, or feature_new3_edge_depth
    else:
        raise ValueError(f"Walk type '{walk}' is not recognized.")

def read_graph(file, path_to_gml, params):

    ac_operator = params['ac_operator']
    model_number = params['model_number']
    depth_max = params['depth_max']
    walk = params['walk']

    # computation dict
    comp_dict = {'MA': np.multiply,
                    'FA': np.multiply,
                    'MD': np.subtract,
                    'FD': np.subtract,
                    'MR': np.divide,
                    'FR': np.divide,
                    'MS': np.add,
                    'FS': np.add}

    # feature list
    feature_set_NBO = vector_feature_NBO(depth_max, ac_operator, model_number, walk)

    # Determine the feature type based on walk and model_number
    feature_type = get_feature_for_walk(walk, feature_set_NBO, model_number)

    file = os.path.join(path_to_gml, file)

    # define the class graph
    G = nx.Graph()
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

    if walk == 'AA':
        if ac_operator in ['MA', 'MD', 'MR', 'MS']:
            AC_vector, AC_vname = atom_atom_MC(G, indx, depth_max, node_dict, feature_set_NBO[0], ac_operator, comp_dict)
        elif ac_operator in ['FA', 'FD', 'FR', 'FS']:
            AC_vector, AC_vname = atom_atom_F(G, depth_max, feature_set_NBO[0], ac_operator, comp_dict)

    elif walk == 'BBavg':
        if ac_operator in ['MA', 'MD', 'MR', 'MS']:
            BB_AC_vector, BB_AC_vname, BB_AC_vector_avg, AC_vname = bond_bond_MC(G, depth_max, edge_dict, feature_set_NBO[1], ac_operator, comp_dict)
    
    elif walk == 'BB':
        if ac_operator in ['MA', 'MD', 'MR', 'MS']:
            BB_AC_vector, AC_vname, BB_AC_vector_avg, BB_AC_vname_avg = bond_bond_MC(G, depth_max, edge_dict, feature_set_NBO[1], ac_operator, comp_dict)
        elif ac_operator in ['FA', 'FD', 'FR', 'FS']:
            AC_vector, AC_vname = bond_bond_F(G, depth_max, feature_set_NBO[1], ac_operator, comp_dict)

    elif walk == 'AB':
        if ac_operator in ['MA', 'MD', 'MR', 'MS']:
            AC_vector, AC_vname = bond_atom_MC(G, indx, depth_max, edge_dict, feature_set_NBO[0], feature_set_NBO[1], ac_operator, comp_dict)
        elif ac_operator in ['FA', 'FD', 'FR', 'FS']:
            AC_vector, AC_vname = bond_atom_F(G, depth_max, feature_set_NBO[0], feature_set_NBO[1], ac_operator, comp_dict)
    
    elif walk == 'ABBAavg':
        if ac_operator in ['MA', 'MD', 'MR', 'MS']:
            AC_vector, AC_vname = new_nbo_bond_bond_MC(G, depth_max, node_dict, edge_dict, model_number, ac_operator, comp_dict)

    elif walk == 'ABBA':
        if ac_operator in ['FA', 'FD', 'FR', 'FS']:
            AC_vector, AC_vname = new_nbo_bond_bond_F(G, depth_max, node_dict, model_number, ac_operator, comp_dict)
 
    else:
        raise ValueError(f"Walk type '{walk}' is not recognized.")
    
    return AC_vname

# Wrapper function for multiprocessing
def process_file(args):
    file, path_to_gml, params = args
    return read_graph(file, path_to_gml, params)

if __name__ == "__main__":

    # time of execution
    start_time = time.time()

    # vector to obtain
    print('Selected NBO AABBA vector:\nOrigin/Operator:', PARAMS['ac_operator'], \
        '  Type of walking:', PARAMS['walk'],  \
        '  Maximum depth:', PARAMS['depth_max'],
        '  Model number (applies for AABBA(II)):',  PARAMS['model_number'] \
    )

    # Determine the directory of the script
    script_dir = Path(__file__).resolve().parent
    
    # Define relative paths based on the script directory
    general_path = script_dir
    
    path_to_gml = general_path / 'uNatQ_graphs'
    path_to_folder = general_path / f'vectors_AABBA/NBO_{PARAMS["walk"]}'

    # Ensure the output directory exists
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    # List comprehension to get all .gml files
    gml_list = [file for file in os.listdir(path_to_gml) if file.endswith('.gml')]

    # save maximum depth
    #get_max_depth(gml_list)

    # Create a process pool and map the files
    with Pool() as pool:
        poolReturn = pool.map(process_file, [(file, path_to_gml, PARAMS) for file in gml_list])

    # Extract the features
    feature_set_NBO = vector_feature_NBO(PARAMS['depth_max'], PARAMS['ac_operator'], PARAMS['model_number'], PARAMS['walk'])
    
    # Determine the features_type selected
    feature_type = get_feature_for_walk(PARAMS['walk'], feature_set_NBO, PARAMS['model_number'])

    # join the vectors
    out_dict = join_vectors(poolReturn, feature_type)

    # save derived vectors in a .csv file
    print('Save vectors in a .csv file')
    save_vectors(path_to_folder, out_dict, PARAMS['depth_max'], PARAMS['ac_operator'], PARAMS['walk'], PARAMS['model_number'])

    # Round features of the CSV
    round_csv(path_to_folder, PARAMS['depth_max'], PARAMS['ac_operator'], PARAMS['walk'], PARAMS['model_number'])

    print("Execution time: " + str(round((time.time() - start_time) / 60, 4)) + " minutes." + str(cpu_count()))