# processing the data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def standarize_train(data_train):

    outlier = []

    for idx, col in enumerate(data_train):
        if data_train[col].std() == 0:
            outlier.append(col)
    print('outlier', outlier)

    data_train = data_train.drop(outlier, axis=1)
    
    mean =  data_train.mean()
    std = data_train.std()
    data_train = (data_train - mean)/std
    data_train = data_train.fillna(0)

    return data_train, mean, std, outlier

def standarize_rest(data, mean, std, outlier):

    data = data.drop(outlier, axis=1)
    data = (data - mean)/std
    data = data.fillna(0)

    return data

def vector_feature_PT(depth_max, ac_type, model_number, walk):

    feature_node = ['feature_atomic_number',
                'feature_identity',
                'feature_node_degree',
                'feature_covalent_radius',
                'feature_electronegativity']

    feature_edge = ['feature_wiberg_bond_order_int',
                    'feature_bond_distance',
                    'feature_identity']

    feature_node_depth, feature_edge_depth = [], []

    # feature heading
    Z =  [f'Z-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    I =  [f'I-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    ND =  [f'T-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    CR =  [f'S-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    X =  [f'chi-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    BO =  [f'BO-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    d =  [f'd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    Zi =  [f'Zi-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]   #_BBavg{model_number}
    Zj =  [f'Zj-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Ti =  [f'Ti-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Tj =  [f'Tj-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Xi =  [f'chi_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Xj =  [f'chi_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Xij =  [f'chi_ij-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Si =  [f'Si-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Sj =  [f'Sj-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    BO_ =  [f'BO-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    d_ =  [f'd-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    I_ =  [f'I-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]

    feature_node_depth = Z  + I + ND + CR + X
    #feature_node_depth.insert(0, 'id')

    feature_edge_depth = BO + d + I
    #feature_edge_depth.insert(0, 'id')

    feature_new1_edge_depth = Zi + Zj + Ti + Tj + Xi + Xj + d_ + BO_ + I_
    #feature_new1_edge_depth.insert(0, 'id')

    feature_new2_edge_depth = Zi + Zj + Ti + Tj + Xij + d_ + BO_ + I_
    #feature_new2_edge_depth.insert(0, 'id')

    feature_new3_edge_depth = Zi + Zj + Ti + Tj + Xij + Si + Sj + BO_ + I_
    #feature_new3_edge_depth.insert(0, 'id')

    feature_set = [feature_node,
                  feature_edge,
                  feature_node_depth,
                  feature_edge_depth,
                  feature_new1_edge_depth,
                  feature_new2_edge_depth,
                  feature_new3_edge_depth]

    return feature_set


def vector_feature_NBO(depth_max, ac_type, model_number, walk):

    feature_node_uNat = ['feature_atomic_number',
                'feature_natural_atomic_charge',
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
    feature_new1_edge_uNat_depth, feature_new2_edge_uNat_depth = [], []
    feature_new4_edge_uNat_depth, feature_new5_edge_uNat_depth = [], []

    # feature heading
    # feature heading for node and edges with NBO properties
    Z =  [f'Z-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    qnat =  [f'qnat-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Vnat =  [f'Vnat-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ns =  [f'Ns-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Np =  [f'Np-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Nd =  [f'Nd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    Nlp =  [f'Nlp-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPe =  [f'LPe-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPocc =  [f'LPocc-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPs =  [f'LPs-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPp =  [f'LPp-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPd =  [f'LPd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LPde =  [f'LPde-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    Nlv =  [f'Nlv-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVe =  [f'LVe-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVocc =  [f'LVocc-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVs =  [f'LVs-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVp =  [f'LVp-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVd =  [f'LVd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    LVde =  [f'LVde-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    BO =  [f'BO-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    d =  [f'd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    Nbn =  [f'Nbn-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNe =  [f'BNe-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNocc =  [f'BNocc-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNs =  [f'BNs-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNp =  [f'BNp-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNd =  [f'BNd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNde =  [f'BNde-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Nbn_ =  [f'Nbn_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNe_ =  [f'BNe_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNocc_ =  [f'BNocc_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNs_ =  [f'BNs_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNp_ =  [f'BNp_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNd_ =  [f'BNd_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    BNde_ =  [f'BNde_-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    E2max =  [f'E2max-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    E2avg =  [f'E2avg-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Dtype =  [f'Dtype-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    De =  [f'De-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Docc =  [f'Docc-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ds =  [f'Ds-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Dp =  [f'Dp-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Dd=  [f'Dd-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Dde =  [f'Dde-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ae =  [f'Ae-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Aocc =  [f'Aocc-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    As =  [f'As-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ap =  [f'Ap-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ad =  [f'Ad-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]
    Ade =  [f'Ade-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    qnat_i =  [f'qnat_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)] #    qnat_i =  [f'qnat_i-{i}_{ac_type}_ABBAavg_{model_number}' for i in range(depth_max + 1)]
    Vnat_i =  [f'Vnat_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Ns_i =  [f'Ns_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Np_i =  [f'Np_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Nd_i =  [f'Nd_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    qnat_j =  [f'qnat_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Vnat_j =  [f'Vnat_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Ns_j =  [f'Ns_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Np_j =  [f'Np_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Nd_j =  [f'Nd_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    
    Nlp_i = [f'Nlp_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Nlp_j = [f'Nlp_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LPe_i = [f'LPe_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LPe_j = [f'LPe_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LPde_i = [f'LPde_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LPde_j = [f'LPde_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]

    Nlv_i =  [f'Nlv_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    Nlv_j =  [f'Nlv_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LVe_i = [f'LVe_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LVe_j = [f'LVe_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LVde_i = [f'LVde_i-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]
    LVde_j = [f'LVde_j-{i}_{ac_type}_ABBAavg' for i in range(depth_max + 1)]

    I =  [f'I-{i}_{ac_type}_{walk}' for i in range(depth_max + 1)]

    feature_node_uNat_depth = Z + qnat + Vnat + Ns + Np + Nd + Nlp + \
        LPde + LPe + LPocc + LPs + LPp + LPd  + Nlv + LVde + LVe + LVocc + \
        LVs + LVp + LVd + I
    #feature_node_uNat_depth.insert(0, 'id')

    feature_node_dNat_depth = Z + qnat + Vnat + Ns + Np + Nd + Nlp + \
        LPde + LPe + LPocc + LPs + LPp + LPd  + Nlv + LVde + LVe + LVocc + \
        LVs + LVp + LVd + I
    #feature_node_dNat_depth.insert(0, 'id')

    feature_edge_uNat_depth = BO + d + Nbn + Nbn_ + BNde + BNe + BNocc + BNs + BNp + \
        BNd + BNde_ + BNe_ +  BNocc_ + BNs_ + BNp_ + BNd_ + I
    #feature_edge_uNat_depth.insert(0, 'id')

    feature_edge_dNat_depth = BO + d + E2max + E2avg + De + Dde + Docc + \
        Ds + Dp + Dd + Ae + Ade + Aocc + As + Ap + Ad + I
    #feature_edge_dNat_depth.insert(0, 'id')

    feature_new1_edge_uNat_depth = qnat_i + qnat_j + Ns_i + Ns_j + \
        Np_i + Np_j + Nd_i +  Nd_j + Nlp_i + Nlp_j + BO + Nbn + BNe + BNe_ + BNs + \
        BNp + BNd + I
    #feature_new1_edge_uNat_depth.insert(0, 'id')

    feature_new2_edge_uNat_depth = Vnat_i + Vnat_j + Ns_i + Ns_j + \
        Np_i + Np_j + Nd_i +  Nd_j + Nlp_i + Nlp_j + BO + Nbn + BNe + BNe_ + BNs + \
        BNp + BNd + I
    #feature_new2_edge_uNat_depth.insert(0, 'id')

    feature_new4_edge_uNat_depth = qnat_i + qnat_j + Vnat_i + Vnat_j + \
        Ns_i + Ns_j + Np_i + Np_j + Nd_i +  Nd_j + Nlp_i + Nlp_j + Nlv_i + Nlv_j + \
        d + BO + Nbn + BNs + BNp + BNd + Nbn_ + BNs_ + BNp_ + BNd_ + I
    #feature_new4_edge_uNat_depth.insert(0, 'id')

    feature_new5_edge_uNat_depth = qnat_i + qnat_j + Vnat_i + Vnat_j + \
        Nlp_i + Nlp_j + LPe_i + LPe_j + LPde_i + LPde_j +  Nlv_i + Nlv_j + \
        LVe_i + LVe_j + LVde_i + LVde_j + d + BO + Nbn + BNe + BNde + Nbn_ + \
        BNe_ + BNde_ + I
    #feature_new5_edge_uNat_depth.insert(0, 'id')

    feature_set = [feature_node_uNat,
                  feature_edge_uNat,
                  feature_edge_dNat,
                  feature_node_uNat_depth,
                  feature_edge_uNat_depth,
                  feature_node_dNat_depth,
                  feature_edge_dNat_depth,
                  feature_new1_edge_uNat_depth,
                  feature_new2_edge_uNat_depth,
                  feature_new4_edge_uNat_depth,
                  feature_new5_edge_uNat_depth
                  ]

    return feature_set

def vector_feature_wholegraph():

    feature_wholegraph = ['feature_charge',
                'feature_molecular_mass',
                'feature_n_atoms',
                'feature_n_electrons']

    feature_wholegraph = []

    # feature heading
    q =  ['feature_charge']
    M =  ['feature_molecular_mass']
    Nat =  ['feature_n_atoms']
    Ne =  ['feature_n_electrons']

    feature_wholegraph = q + M + Nat + Ne

    return feature_wholegraph
