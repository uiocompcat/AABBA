from cmath import isinf, nan
import networkx as nx # networkx to read and visualize
import numpy as np
from numpy import inf
from graph_info import node_info, edge_info, new_edge_info, gp_new_edge_attribute, nbo_new_edge_attribute


def atom_atom_MC(G, idx, depth_max, node_dict, feature_node, ac_operator, comp):
    """Apply the metal-centered atom-atom autocorrelation function ("MC")

    Args:
        G (networkx graph class): Graph read by networkx
        idx (int): Node index set as the starting node
        depth_max (int): Maximum distance to read the graph
        node_dict (dict): Depth as keys and nodes as values
        feature_node (list): List with all the node attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation AA vector at a specified depth (MC)
    """
    ac_vector = []

    for feature in feature_node:

        idx_feature = G.nodes[str(idx)][feature]

        # search for any index property equals to cero
        if idx_feature == 0:
            pass
            #print('feature: ', feature, 'id',  G.graph['meta_data']['id'] )
        else:
            pass

        # set as cero the division of a number by cero
        np.seterr(invalid='ignore')

        # compute and save autocorrelation product at depth = 0
        out_idx = comp[f'{ac_operator}'](float(idx_feature), float(idx_feature))

        # remove NaN product
        if np.isnan(out_idx) or np.isinf(out_idx):
            out_idx = 0
        ac_vector.append(out_idx)

        for depth in range(depth_max + 1):

            feature_value = 0
            # compute autocorrelation product at depth > 0
            if depth != 0:

                for node in node_dict[depth]:

                    node_feature = G.nodes[node][feature]

                    # set as cero the division of a number by cero
                    np.seterr(invalid='ignore')
                    item = comp[f'{ac_operator}'](float(idx_feature), float(node_feature))

                    feature_value = feature_value + item

                # remove NaN product
                if np.isinf(feature_value) or np.isnan(feature_value):
                    feature_value = 0

                # save autocorrelation vector at specified depth
                ac_vector.append(feature_value)
                ac_name = ac_vector.copy()

    ac_name.insert(0, G.graph['meta_data']['id'])

    return ac_vector, ac_name

def atom_atom_F(G, depth_max, feature_node, ac_operator, comp):
    """Apply the full atom-atom autocorrelation function ("F")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        feature_node (list): List with all the node attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Full autocorrelation vector AA at a specified depth (F)
    """

    full_ac, full_ac_vector = [], []

    for node_str in G.nodes():

        # select each node index as starting node
        node = int(node_str)

        node_dict = node_info(G, depth_max, node)

        # compute and save autocorrelation products at specified depths
        ac_vector, ac_name = atom_atom_MC(G, node, depth_max, node_dict, feature_node, ac_operator, comp)

        full_ac.append(ac_vector)

    for ind_vector in zip(*full_ac):

        # sum the all vectors derived from each starting node
        #full_vector = sum(list(ind_vector))
        #full_ac_vector.append(full_vector)
        full_ac_vector.append(sum(list(ind_vector)))

        full_ac_name = full_ac_vector.copy()

    full_ac_name.insert(0, G.graph['meta_data']['id'])

    return full_ac_vector, full_ac_name

def bond_bond_MC(G, depth_max, edge_dict, feature_edge, ac_operator, comp):
    """Calculate the bond-bond autocorrelation vector for a single edge ("MC")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        edge_dict (dict): Depth as keys and nodes as values
        feature_edge (list): List with all the edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation BB vector at a specified depth  (MC)
    """
    ac_vector, ac_vector_avg = [], []

    for feature in feature_edge:

        # define the super bond as the sum of all the edges at depth = 0
        super_bond = 0
        for edge in edge_dict[0]:
            idx_feature = G.edges[edge][feature]
            super_bond = super_bond + idx_feature

        # search for any index property equals to cero
        if idx_feature == 0:
            pass
            #print('feature: ', feature, 'id',  G.graph['meta_data']['id'] )
        else:
            pass

        # set as cero the division of a number by cero
        np.seterr(invalid='ignore')

        # define the average bond as the average of all the edges at depth = 0
        average_bond = np.divide(super_bond, len(edge_dict[0]))

        # compute and save autocorrelation product at depth = 0
        out_idx = comp[f'{ac_operator}'](float(super_bond), float(super_bond))
        out_avg_idx = comp[f'{ac_operator}'](float(average_bond), float(average_bond))

        # remove NaN product
        if np.isnan(out_idx) or np.isinf(out_idx):
            out_idx = 0
        ac_vector.append(out_idx)
        if np.isnan(out_avg_idx) or np.isinf(out_avg_idx):
            out_avg_idx = 0
        ac_vector_avg.append(out_avg_idx)

        for depth in range(depth_max + 1):

            feature_value = 0
            feature_value_avg = 0

            # compute autocorrelation product at depth > 0
            if depth != 0:

                for edge in edge_dict[depth]:

                    edge_feature = G.edges[edge][feature]

                    item = comp[f'{ac_operator}'](float(super_bond), float(edge_feature))
                    item_average = comp[f'{ac_operator}'](float(average_bond), float(edge_feature))

                    feature_value = feature_value + item
                    feature_value_avg = feature_value_avg + item_average

                # remove NaN product
                if np.isinf(feature_value) or np.isnan(feature_value):
                    feature_value = 0
                if np.isinf(feature_value_avg) or np.isnan(feature_value_avg):
                    feature_value_avg = 0

                # save autocorrelation vector at specified depth
                ac_vector.append(feature_value)
                ac_vector_avg.append(feature_value_avg)

                ac_name = ac_vector.copy()
                ac_avg_name = ac_vector_avg.copy()

    ac_name.insert(0, G.graph['meta_data']['id'])
    ac_avg_name.insert(0, G.graph['meta_data']['id'])

    return ac_vector, ac_name, ac_vector_avg, ac_avg_name

def bond_bond(G, idx_edge, depth_max, edge_dict, features, ac_operator, comp):
    """Apply the bond-bond autocorrelation function starting for a single edge

    Args:
        G (networkx graph class): Graph read by networkx
        idx (list): Edge index set as the starting edge
        depth_max (int): Maximum distance to read the graph
        edge_dict (dict): Depth as keys and edges as values
        features (list): List of attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation BB vector at a specified depth
    """

    ac_vector = []

    for feature in features:

        idx_edge = (idx_edge[0], idx_edge[1], 0)

        idx_feature = G.edges[idx_edge][feature]

        # search for any index property equals to cero
        if idx_feature == 0:
            pass
            #print('feature: ', feature, 'id',  G.graph['meta_data']['id'] )
        else:
            pass

        # set as cero the division of a number by cero
        np.seterr(invalid='ignore')

        # compute and save autocorrelation product at depth = 0
        out_idx = comp[f'{ac_operator}'](float(idx_feature), float(idx_feature))

        # remove NaN product
        if np.isnan(out_idx) or np.isinf(out_idx):
            out_idx = 0
        ac_vector.append(out_idx)

        for depth in range(depth_max + 1):

            feature_value = 0

            # compute autocorrelation product at depth > 0
            if depth != 0:

                for edge in edge_dict[depth]:

                    edge_feature = G.edges[edge][feature]
                    item = comp[f'{ac_operator}'](float(idx_feature), float(edge_feature))
                    feature_value = feature_value + item

                # remove NaN product
                if np.isnan(feature_value) or np.isinf(feature_value):
                    feature_value = 0

                # save autocorrelation vector at specified depth
                ac_vector.append(feature_value)

                ac_name = ac_vector.copy()

    ac_name.insert(0, G.graph['meta_data']['id'])

    return ac_vector, ac_name

def bond_bond_F(G, depth_max, feature_edge, ac_operator, comp):
    """Apply the full bond-bond autocorrelation function ("F")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        feature_edge (str): Edge attribute to be used
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Full autocorrelation vector BB at a specified depth (F)
    """
    full_ac, full_ac_vector = [], []

    for edge in G.edges():

        # select each edge index as starting edge
        edge_dict = new_edge_info(G, depth_max, edge)

        # compute and save autocorrelation products at specified depths
        ac_vector, ac_name = bond_bond(G, edge, depth_max, edge_dict, feature_edge, ac_operator, comp)

        full_ac.append(ac_vector)

    for ind_vector in zip(*full_ac):

        # sum the all vectors derived from each edge
        #full_vector = sum(list(ind_vector))
        #full_ac_vector.append(full_vector)
        full_ac_vector.append(sum(list(ind_vector)))

        full_ac_name = full_ac_vector.copy()

    full_ac_name.insert(0, G.graph['meta_data']['id'])

    return full_ac_vector, full_ac_name

def bond_atom_MC(G, idx, depth_max, edge_dict, feature_node, feature_edge, ac_operator, comp):
    """Apply the atom-bond autocorrelation function ("AB")

    Args:
        G (networkx graph class): Graph read by networkx that contains the nodes
        idx (int): Node index set as the starting node
        depth_max (int): Maximum depth to read the graph
        edge_dict (dict): Depth as keys and edges as values
        feature_node (list): List of node attributes
        feature_edge (list): List of edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation BA vector at a specified depth  (MC)
    """
    ac_vector = []

    for feature_n in feature_node:

        idx_feature = G.nodes[str(idx)][feature_n]

        for depth in range(depth_max + 1):

            feature_prods = []

            for feature_e in feature_edge:

                feature_value = 0

                for edge in edge_dict[depth]:

                    # compute autocorrelation product at specified depth
                    edge_feature = G.edges[edge][feature_e]
                    item = comp[f'{ac_operator}'](float(idx_feature), float(edge_feature))
                    feature_value = feature_value + item

                # remove NaN product
                if np.isnan(feature_value) or np.isinf(feature_value):
                    feature_value = 0

                feature_prods.append(feature_value)
                feature_sum = sum(map(float, feature_prods))

            # save autocorrelation product at specified depth
            ac_vector.append(feature_sum)
            ac_name = ac_vector.copy()

    ac_name.insert(0, G.graph['meta_data']['id'])

    return ac_vector, ac_name

def bond_atom_F(G, depth_max, feature_node, feature_edge, ac_operator, comp):
    """Apply the full bond-atom autocorrelation function ("F")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum depth to read the graph
        feature_node (list): List of node attributes
        feature_edge (list): List of edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Full autocorrelation vector BA at a specified depth (F)
    """
    full_ac, full_ac_vector = [], []

    for node_str in G.nodes():

        # select each node index as starting node
        node = int(node_str)

        edge_dict = edge_info(G, depth_max, node)

        # compute and save autocorrelation products at specified depths
        ac_vector, ac_name = bond_atom_MC(G, node, depth_max, edge_dict, feature_node, feature_edge, ac_operator, comp)

        full_ac.append(ac_vector)

    for ind_vector in zip(*full_ac):

        # sum the all vectors derived from each starting node
        full_vector = sum(list(ind_vector))
        full_ac_vector.append(full_vector)

        full_ac_name = full_ac_vector.copy()

    full_ac_name.insert(0, G.graph['meta_data']['id'])

    return full_ac_vector, full_ac_name

def new_gp_bond_bond_MC(G, depth_max, node_dict, edge_dict, model_number, ac_operator, comp):
    """Apply the new-bond-bond nBB autocorrelation function ("MC")
 
    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum depth to read the graph
        node_dict (dict): Depth as keys and nodes as values
        edge_dict (dict): Depth as keys and edges as values
        model_number (int): Type of new edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation vector nBB at a specified depth (MC)
    """

    # add new attributes to the edge entities
    new_feature_edge = gp_new_edge_attribute(G, model_number, node_dict)

    # compute the bond_bond autocorrelation function with the new atttributes
    ac_vector, ac_name, ac_vector_avg, ac_avg_name = bond_bond_MC(G, depth_max, edge_dict, new_feature_edge, ac_operator, comp)

    return ac_vector_avg, ac_avg_name

def new_nbo_bond_bond_MC(G, depth_max, node_dict, edge_dict, model_number, ac_operator, comp):
    """Apply the new-bond-bond nBB autocorrelation function ("MC")
 
    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum depth to read the graph
        node_dict (dict): Depth as keys and nodes as values
        edge_dict (dict): Depth as keys and edges as values
        model_number (int): Type of new edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties

    Returns:
        list: Autocorrelation vector nBB at a specified depth (MC)
    """

    # add new attributes to the edge entities
    new_feature_edge = nbo_new_edge_attribute(G, model_number, node_dict)

    # compute the bond_bond autocorrelation function with the new atttributes
    ac_vector, ac_name, ac_vector_avg, ac_avg_name = bond_bond_MC(G, depth_max, edge_dict, new_feature_edge, ac_operator, comp)

    return ac_vector_avg, ac_avg_name

def new_gp_bond_bond_F(G, depth_max, node_dict, model, ac_operator, comp):
    """Apply the full new-bond-bond nBB autocorrelation function ("F")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum depth to read the graph
        node_dict (dict): Depth as keys and nodes as values
        model_number (int): Type of new edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties


    Returns:
        list: Full autocorrelation vector nBB at a specified depth (F)
    """

    full_ac, full_ac_vector = [], []

    new_edge_features = gp_new_edge_attribute(G, model, node_dict)

    for edge in G.edges():

        # select each edge index as starting edge
        edge_dict = new_edge_info(G, depth_max, edge)

        # compute and save autocorrelation products at specified depths
        ac_vector, ac_name = bond_bond(G, edge, depth_max,  edge_dict, new_edge_features, ac_operator, comp)

        full_ac.append(ac_vector)

    for ind_vector in zip(*full_ac):

        # sum the all vectors derived from each edge
        #full_vector = sum(list(ind_vector))
        #full_ac_vector.append(full_vector)
        full_ac_vector.append(sum(list(ind_vector)))

        full_ac_name = full_ac_vector.copy()

    full_ac_name.insert(0, G.graph['meta_data']['id'])

    return full_ac_vector, full_ac_name

def new_nbo_bond_bond_F(G, depth_max, node_dict, model, ac_operator, comp):
    """Apply the full new-nbo-bond-bond nBB autocorrelation function ("F")

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum depth to read the graph
        node_dict (dict): Depth as keys and nodes as values
        model_number (int): Type of new edge attributes
        ac_operator (str): Arithmetic operator applied to the properties
        comp (dict): Arithmetic operator applied to the properties


    Returns:
        list: Full autocorrelation vector nbo nBB at a specified depth (F)
    """

    full_ac, full_ac_vector = [], []

    new_edge_features = nbo_new_edge_attribute(G, model, node_dict)

    for edge in G.edges():

        # select each edge index as starting edge
        edge_dict = new_edge_info(G, depth_max, edge)

        # compute and save autocorrelation products at specified depths
        ac_vector, ac_name = bond_bond(G, edge, depth_max,  edge_dict, new_edge_features, ac_operator, comp)

        full_ac.append(ac_vector)

    for ind_vector in zip(*full_ac):

        # sum the all vectors derived from each edge
        full_ac_vector.append(sum(list(ind_vector)))
        full_ac_name = full_ac_vector.copy()

    full_ac_name.insert(0, G.graph['meta_data']['id'])

    return full_ac_vector, full_ac_name