from operator import index
import networkx as nx
import numpy as np


def metal_index(G):
    """Identify the metal center in the graph by its atomic number. Provide
        its node index.

    Args:
        G (networkx graph class): Graph read by networkx

    Returns:
        int: metal node index
    """

    # metal atomic numbers
    period_4 = [i for i in range(21,31)]
    period_5 = [i for i in range(39, 49)]
    period_6 = [i for i in range(57,58)] + [i for i in range(72, 81)]
    period_7 = [i for i in range(89,90)] + [i for i in range(104,109)]
    metals = [*period_4, *period_5, *period_6, *period_7]

    atomic_numbers = []

    for node in G.nodes():

        # save the atomic numbers of the graph
        atomic_numbers.append(G.nodes[node]["feature_atomic_number"])

    # identify the metal atomic number of the graph
    metal_number = [atomic_num for atomic_num in atomic_numbers
                    if atomic_num in metals]

    # find the metal node index
    idx_metal = atomic_numbers.index(metal_number[0])

    return idx_metal

def node_info(G, depth_max, idx):
    """Create a dictionary for each graph where the keys are the depths and,
        the values are the nodes in those depths

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        idx (int): Node index set as the starting node

    Returns:
        dict: per each graph at the selected depth

    """
    # save the already read nodes index per iteration
    historical_nodes = []
    historical_nodes.append(str(idx))

    # create a dict at depth = 0
    node_dict = dict()
    origin = 0
    keys = origin
    values = [str(idx)]
    node_dict[keys] = values

    for depth in range(depth_max):

        # save the node indexes at depth = 1
        if depth == 0:
            depth = depth + 1

            node_neighbors_idx = [n for n in G.neighbors(str(idx))]
            node_dict[depth] = node_neighbors_idx

            historical_nodes.extend(node_neighbors_idx)

        # save the node indexes at depth > 1
        else:
            node_neighbors_depth = []

            for node in node_dict[depth]:

                node_neighbors = [n for n in G.neighbors(node) \
                                    if n not in historical_nodes]
                node_neighbors_depth.extend(node_neighbors)

                historical_nodes.extend(node_neighbors)

            node_dict[depth + 1] = list(set(node_neighbors_depth))

            depth = depth + 1

    return node_dict

def edge_info(G, depth_max, idx):
    """Create a dictionary for each graph where the keys are the depths and,
        the values are the edges in those depths starting from one new defined edge

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        index (int): Node index set as the starting node

    Returns:
        dict: per each graph at the selected depth

    """
    # save the already read node indexes per iteration
    historical_nodes = []
    historical_nodes.append(str(idx))

    edge_dict = dict()

    for depth in range(depth_max + 1):

        # save the edge index at depth = 0
        if depth == 0:

            node_neighbors = []

            edge_neighbors_idx = [G.edges(str(idx))]
            edges_idx = [(node1, node2) for edge in edge_neighbors_idx
                                for node1, node2 in edge]
            edges_format_idx = [(node1, node2, 0) for edge in edge_neighbors_idx
                                for node1, node2 in edge]

            edge_dict[depth] = edges_format_idx

            for edge in edges_idx:

                # save the node indexes of the edge except the starting node index
                node1, node2 = edge

                if node1 != str(idx):
                    node_neighbors.append(node1)
                if node2 != str(idx):
                    node_neighbors.append(node2)
                node_neighbors = list(set(node_neighbors))

        # save the edge indexes at depth > 0
        else:
            edges_depth, edges_format_depth = [], []

            for node in node_neighbors:

                edge_neighbors = [G.edges(node)]

                for edge in edge_neighbors:

                    for node1, node2 in edge:

                        if node1 not in historical_nodes and node1 != node2:
                            if node2 not in historical_nodes and node2 != node1:

                                new_edge = (node1, node2)
                                new_edge_format = (node1, node2, 0)

                                edges_depth.append(new_edge)
                                edges_format_depth.append(new_edge_format)

                historical_nodes.append(node)

            edge_dict[depth] = edges_format_depth

            # save the non-walked node indexes yet
            active_nodes = []

            for edge in edges_depth:

                node1, node2 = edge

                if node1 not in historical_nodes:
                    active_nodes.append(node1)
                if node2 not in historical_nodes:
                    active_nodes.append(node2)

            node_neighbors = active_nodes
            depth = depth + 1

    return edge_dict

def new_edge_info(G, depth_max, idx_edge):
    """Create a dictionary for each graph where the keys are the depths and,
        the values are the edges in those depths starting from a specific idx_edge

    Args:
        G (networkx graph class): Graph read by networkx
        depth_max (int): Maximum distance to read the graph
        idx_edge (tuple): Edge index set as the starting edge

    Returns:
        dict: per each graph at the selected depth

    """
    # save the starting edge index
    (node1_idx, node2_idx) = idx_edge
    idx_edge_rev = (node2_idx, node1_idx)

    historical_edges = [idx_edge]
    historical_edges.append(idx_edge_rev)

    historical_nodes = []

    # create a dict at depth = 0
    edge_dict = dict()
    origin = 0
    keys = origin
    values = (node1_idx, node2_idx, 0)
    edge_dict[keys] = values

    for depth in range(depth_max):

        # save the edge indexes at depth = 1
        if depth == 0:
            depth = depth + 1

            node1_idx, node2_idx = idx_edge

            historical_nodes = [node1_idx]
            historical_nodes.append(node2_idx)

            # save the neighbor edge indexes of the starting edge
            edge_neighbors_idx_1 = [n for n in G.edges(node1_idx)]
            edge_neighbors_idx_2 = [n for n in G.edges(node2_idx)]
            edge_neighbors_idx_sum = edge_neighbors_idx_1 + edge_neighbors_idx_2

            edge_neighbors_idx = [edge for edge in edge_neighbors_idx_sum
                                if edge not in historical_edges]

            edges_format_idx = []
            node_neighbors = []

            for edge in edge_neighbors_idx:

                # save the read edge indexes
                node1, node2 = edge
                edges_format_idx.append((node1, node2, 0))

                if node1 not in historical_nodes:
                    node_neighbors.append(node1)
                if node2 not in historical_nodes:
                    node_neighbors.append(node2)
                node_neighbors = list(set(node_neighbors))

                edge_rev = (node2, node1)
                historical_edges.append(edge)
                historical_edges.append(edge_rev)

            edge_dict[depth] = edges_format_idx

        # save the edges at depth > 1
        else:
            historical_nodes.extend(node_neighbors)

            edge_neighbors = [edge for edge in G.edges(node_neighbors)
                            if edge not in historical_edges]

            edges_format = []
            active_nodes = []

            for edge in edge_neighbors:

                # save the read edge indexes
                node1, node2 = edge
                edges_format.append((node1, node2, 0))

                # save the non-walked node indexes yet
                if node1 not in historical_nodes:
                    active_nodes.append(node1)
                if node2 not in historical_nodes:
                    active_nodes.append(node2)
                active_nodes = list(set(active_nodes))

                edge_rev = (node2, node1)
                historical_edges.append(edge)
                historical_edges.append(edge_rev)

            edge_dict[depth + 1] = edges_format

            node_neighbors = active_nodes
            depth = depth + 1

    return edge_dict

def new_edge_attribute(G, model_number, node_dict):
    """Define new edge attributes according to the selected model

    Args:
        G (networkx graph class): Graph read by networkx
        model_number (int): Type of new edge attribute
        node_dict (dict): Depth as keys and nodes as values

    Returns:
        list: List of new edge features
    """
    # dictionary for new attributes
    for edge in G.edges(G):

        # sort nodes of in the edges by their depths (in node_dict)
        node_i, node_j = edge

        depth_a = [k for (k, v) in node_dict.items() if node_i in v]
        depth_b = [k for (k, v) in node_dict.items() if node_j in v]

        if depth_a < depth_b:
            edge = node_i, node_j
        if depth_b < depth_a:
            edge = node_j, node_i

        node1, node2 = edge

        # get the new edge features
        Z_1 = G.nodes[node1]['feature_atomic_number']
        Z_2 = G.nodes[node2]['feature_atomic_number']
        T_1 = G.nodes[node1]['feature_node_degree']
        T_2 = G.nodes[node2]['feature_node_degree']
        S_1 = G.nodes[node1]['feature_covalent_radius']
        S_2 = G.nodes[node2]['feature_covalent_radius']
        X_1 = G.nodes[node1]['feature_electronegativity']
        X_2 = G.nodes[node2]['feature_electronegativity']

        # add new edge attributes
        if model_number == 1:
            new_attributes = {(node1, node2, 0): {"feature_atomic_number_1": Z_1,
                                    "feature_atomic_number_2": Z_2,
                                    "feature_node_degree_1": T_1,
                                    "feature_node_degree_2": T_2,
                                    "feature_electronegativity_1": X_1,
                                    "feature_electronegativity_2": X_2}}

            newedge_attribute = ["feature_atomic_number_1",
                                "feature_atomic_number_2",
                                "feature_node_degree_1",
                                "feature_node_degree_2",
                                "feature_electronegativity_1",
                                "feature_electronegativity_2",
                                "feature_bond_distance",
                                "feature_wiberg_bond_order_int",
                                "feature_identity"]

        if model_number == 2:
            new_attributes = {(node1, node2, 0): {"feature_atomic_number_1": Z_1,
                                    "feature_atomic_number_2": Z_2,
                                    "feature_node_degree_1": T_1,
                                    "feature_node_degree_2": T_2,
                                    "feature_electronegativity_12": X_1 - X_2}}

            newedge_attribute = ["feature_atomic_number_1",
                                "feature_atomic_number_2",
                                "feature_node_degree_1",
                                "feature_node_degree_2",
                                "feature_electronegativity_12",
                                "feature_bond_distance",
                                "feature_wiberg_bond_order_int",
                                "feature_identity"]

        if model_number == 3:
            new_attributes = {(node1, node2, 0): {"feature_atomic_number_1": Z_1,
                                    "feature_atomic_number_2": Z_2,
                                    "feature_node_degree_1": T_1,
                                    "feature_node_degree_2": T_2,
                                    "feature_electronegativity_12": X_1 - X_2,
                                    "feature_covalent_radius_1": S_1,
                                    "feature_covalent_radius_2":S_2}}

            newedge_attribute = ["feature_atomic_number_1",
                                "feature_atomic_number_2",
                                "feature_node_degree_1",
                                "feature_node_degree_2",
                                "feature_electronegativity_12",
                                "feature_covalent_radius_1",
                                "feature_covalent_radius_2",
                                "feature_wiberg_bond_order_int",
                                "feature_identity"]

        # add new edge attributes
        nx.set_edge_attributes(G, new_attributes)

    return newedge_attribute

def vector_feature_PT(depth_max, ac_operator, model_number, walk):
    """Generate a set of labels for the PT features at different depths

    Args:
        depth_max (int): Maximum distance to read the graph
        ac_operator (str): Arithmetic operator applied to the properties
        model_number (int): Type of new edge attribute
        walk (str): Type of autocorrelation to be performed

    Returns:
        list: Labels to define the AC vector with PT components
    """
    # set of attributes for nodes and edges
    feature_node = ['feature_atomic_number',
                'feature_identity',
                'feature_node_degree',
                'feature_covalent_radius',
                'feature_electronegativity']

    feature_edge = ['feature_wiberg_bond_order_int',
                    'feature_bond_distance',
                    'feature_identity']

    feature_node_depth, feature_edge_depth = [], []

    # feature heading for node and edges with PT properties
    Z =  [f'Z-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    I =  [f'I-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    ND =  [f'T-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    CR =  [f'S-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    X =  [f'chi-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    BO =  [f'BO-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    d =  [f'd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    Zi =  [f'Zi-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Zj =  [f'Zj-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Ti =  [f'Ti-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Tj =  [f'Tj-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Xi =  [f'chi_i-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Xj =  [f'chi_j-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Xij =  [f'chi_ij-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Si =  [f'Si-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    Sj =  [f'Sj-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    BO_ =  [f'BO-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    d_ =  [f'd-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]
    I_ =  [f'I-{i}_{ac_operator}_BB{model_number}' for i in range(depth_max + 1)]

    feature_node_depth = Z  + I + ND + CR + X
    feature_node_depth.insert(0, 'id')

    feature_edge_depth = BO + d + I
    feature_edge_depth.insert(0, 'id')

    feature_new1_edge_depth = Zi + Zj + Ti + Tj + Xi + Xj + d_ + BO_ + I_
    feature_new1_edge_depth.insert(0, 'id')

    feature_new2_edge_depth = Zi + Zj + Ti + Tj + Xij + d_ + BO_ + I_
    feature_new2_edge_depth.insert(0, 'id')

    feature_new3_edge_depth = Zi + Zj + Ti + Tj + Xij + Si + Sj + BO_ + I_
    feature_new3_edge_depth.insert(0, 'id')

    feature_set = [feature_node,
                  feature_edge,
                  feature_node_depth,
                  feature_edge_depth,
                  feature_new1_edge_depth,
                  feature_new2_edge_depth,
                  feature_new3_edge_depth]

    return feature_set

def vector_feature_NBO(depth_max, ac_operator, walk):
    """Generate a set of labels for the NBO features at different depths

    Args:
        depth_max (int): Maximum distance to read the graph
        ac_operator (str): Arithmetic operator applied to the properties
        walk (str): Type of autocorrelation to be performed

    Returns:
        list: Labels to define the AC vector with NBO components
    """
    # set of attributes for nodes and edges
    feature_node_uNat = ['feature_natural_atomic_charge',
                'feature_natural_electron_population_valence',
                'feature_natural_electron_configuration_0',
                'feature_natural_electron_configuration_1',
                'feature_natural_electron_configuration_2',
                'feature_n_lone_pairs',
                'feature_lone_pair_energy_min_max_difference',
                'feature_lone_pair_max_energy',
                'feature_lone_pair_max_occupation',
                'feature_lone_pair_max_0',
                'feature_lone_pair_max_1',
                'feature_lone_pair_max_2',
                'feature_n_lone_vacancies',
                'feature_lone_vacancy_energy_min_max_difference',
                'feature_lone_vacancy_min_energy',
                'feature_lone_vacancy_min_occupation',
                'feature_lone_vacancy_min_0',
                'feature_lone_vacancy_min_1',
                'feature_lone_vacancy_min_2',
                'feature_identity'
                ]

    feature_edge_uNat = ['feature_wiberg_bond_order',
                'feature_bond_distance',
                'feature_n_bn',
                'feature_n_nbn',
                'feature_bond_energy_min_max_difference',
                'feature_bond_max_energy',
                'feature_bond_max_occupation',
                'feature_bond_max_0',
                'feature_bond_max_1',
                'feature_bond_max_2',
                'feature_antibond_energy_min_max_difference',
                'feature_antibond_min_energy',
                'feature_antibond_min_occupation',
                'feature_antibond_min_0',
                'feature_antibond_min_1',
                'feature_antibond_min_2',
                'feature_identity']

    feature_edge_dNat = ['feature_wiberg_bond_order',
                'feature_bond_distance',
                'feature_stabilisation_energy_max',
                'feature_stabilisation_energy_average',
                'feature_donor_nbo_energy',
                'feature_donor_nbo_min_max_energy_gap',
                'feature_donor_nbo_occupation',
                'feature_donor_nbo_0',
                'feature_donor_nbo_1',
                'feature_donor_nbo_2',
                'feature_acceptor_nbo_energy',
                'feature_acceptor_nbo_min_max_energy_gap',
                'feature_acceptor_nbo_occupation',
                'feature_acceptor_nbo_0',
                'feature_acceptor_nbo_1',
                'feature_acceptor_nbo_2',
                'feature_identity']

    feature_node_uNat_depth, feature_edge_uNat_depth = [], []
    feature_node_dNat_depth, feature_edge_dNat_depth = [], []

    # feature heading for node and edges with NBO properties
    #Z =  [f'Z-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    qnat =  [f'qnat-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Vnat =  [f'Vnat-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ns =  [f'Ns-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Np =  [f'Np-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Nd =  [f'Nd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    Nlp =  [f'Nlp-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPe =  [f'LPe-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPocc =  [f'LPocc-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPs =  [f'LPs-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPp =  [f'LPp-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPd =  [f'LPd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LPde =  [f'LPde-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    Nlv =  [f'Nlv-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVe =  [f'LVe-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVocc =  [f'LVocc-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVs =  [f'LVs-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVp =  [f'LVp-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVd =  [f'LVd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    LVde =  [f'LVde-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    BO =  [f'BO-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    d =  [f'd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    Nbn =  [f'Nbn-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNe =  [f'BNe-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNocc =  [f'BNocc-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNs =  [f'BNs-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNp =  [f'BNp-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNd =  [f'BNd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNde =  [f'BNde-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Nbn_ =  [f'Nbn_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNe_ =  [f'BNe_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNocc_ =  [f'BNocc_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNs_ =  [f'BNs_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNp_ =  [f'BNp_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNd_ =  [f'BNd_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    BNde_ =  [f'BNde_-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    E2max =  [f'E2max-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    E2avg =  [f'E2avg-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Dtype =  [f'Dtype-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    De =  [f'De-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Docc =  [f'Docc-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ds =  [f'Ds-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Dp =  [f'Dp-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Dd=  [f'Dd-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Dde =  [f'Dde-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ae =  [f'Ae-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Aocc =  [f'Aocc-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    As =  [f'As-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ap =  [f'Ap-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ad =  [f'Ad-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]
    Ade =  [f'Ade-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    I =  [f'I-{i}_{ac_operator}_{walk}' for i in range(depth_max + 1)]

    feature_node_uNat_depth = qnat + Vnat + Ns + Np + Nd + Nlp + \
        LPde + LPe + LPocc + LPs + LPp + LPd  + Nlv + LVde + LVe + LVocc + \
        LVs + LVp + LVd + I
    feature_node_uNat_depth.insert(0, 'id')

    feature_node_dNat_depth = qnat + Vnat + Ns + Np + Nd + Nlp + \
        LPde + LPe + LPocc + LPs + LPp + LPd  + Nlv + LVde + LVe + LVocc + \
        LVs + LVp + LVd + I
    feature_node_dNat_depth.insert(0, 'id')

    feature_edge_uNat_depth = BO + d + Nbn + Nbn_ + BNde + BNe + BNocc + BNs + BNp + \
        BNd + BNde_ + BNe_ +  BNocc_ + BNs_ + BNp_ + BNd_ + I
    feature_edge_uNat_depth.insert(0, 'id')

    feature_edge_dNat_depth = BO + d + E2max + E2avg + De + Dde + Docc + \
        Ds + Dp + Dd + Ae + Ade + Aocc + As + Ap + Ad + I
    feature_edge_dNat_depth.insert(0, 'id')

    feature_set = [feature_node_uNat,
                  feature_edge_uNat,
                  feature_edge_dNat,
                  feature_node_uNat_depth,
                  feature_edge_uNat_depth,
                  feature_node_dNat_depth,
                  feature_edge_dNat_depth]

    return feature_set
