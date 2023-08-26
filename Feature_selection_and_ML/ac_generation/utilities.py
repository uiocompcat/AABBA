import numpy as np
import csv
import pandas as pd

def round_numbers(value, feature):
    """Round the values according to their feature

    Args:
        feature (str): Attribute name
        value (float): Value of the attribute to be rounded

    Returns:
        value_out: Rounded attribute value
    """

    if feature == 'feature_atomic_number' or 'feature_identity' or \
                'feature_node_degree' or 'feature_atomic_number1' or \
                'feature_atomic_number2' or 'feature_node_degree1' or \
                'feature_node_degree2':
        value_out = np.round(value, decimals= 1)

    if feature == 'feature_covalent_radius' or 'feature_electronegativity' or \
                'feature_covalent_radius1' or 'feature_covalent_radius2' or \
                'feature_electronegativity1' or 'feature_electronegativity2':
        value_out = np.round(value, decimals = 3)

    return value_out

def join_vectors(vectors, feature):
    """Join all the autocorrelation vector of each graph in a list of dict

    Args:
        vectors (list): AC vector of each graph
        feature (list[str]): Labels to define the elements of the AC vector

    Returns:
        list: List of dict with the AC vector of each graph
    """
    print('Pack the vectors')
    out_dict = []

    for values in vectors:

        keys = feature
        outfile = dict((k, v) for (k, v) in zip(keys, values))
        out_dict.append(outfile)

    return out_dict

def round_csv(path, depth_max, ac_operator, walk):
    """Round the values of the csv file according to their feature

    Args:
        path (path): Path to .csv file folder
        depth_max (int): Maximum distance to read the graph
        ac_operator (str): Arithmetic operator applied to the properties
        walk (str): Type of autocorrelation to be performed
    """
    df = pd.read_csv(path/f'{ac_operator}_{walk}_d{depth_max}.csv')

    # feature heading for node and edges PT and NBO properties
    one_decimal = ('Z-', 'I-', 'T-', 'Zi-', 'Zj-', 'Ti-', 'Tj-', 'Nlp-', \
        'Nlv-', 'Nbn-', 'Nbn_-')
    three_decimal = ('S', 'chi', 'Si', 'Sj', 'chi_i', 'chi_j', 'Ns', 'Np', \
        'Nd' )
    five_decimal = ('BO-')
    six_decimal = ('qnat-', 'Vnat-','LPde-', 'LPe-', 'LPocc-', 'LPs-', \
        'LPs-', 'LPp-', 'LPd-', 'LVde-', 'LVe-', 'LVocc-', 'LVs-', 'LVp-', 'LVd-', \
        'BNde-','BNe-', 'BNocc-', 'BNs-', 'BNp-', 'BNd-', 'BNde_-', 'BNe_-', \
        'BNocc_-', 'BNs_-', 'BNp_-', 'BNd_-')
    seven_decimal = ('d-' , \
                'Davg-', 'Aavg-')

    for col in df.columns:
        if col.startswith(one_decimal):
            df[col] = df[col].round(1)
        if col.startswith(three_decimal):
            df[col] = df[col].round(3)
        if col.startswith(five_decimal):
            df[col] = df[col].round(5)
        if col.startswith(six_decimal):
            df[col] = df[col].round(6)
        if col.startswith(seven_decimal):
            df[col] = df[col].round(7)

    # save the rounded csv file
    df.to_csv(path/f'{ac_operator}_{walk}_d{depth_max}.csv', index=False)

def save_vectors(path, vectors, depth_max, ac_operator, walk):
    """Save the list of vectors in a .csv file

    Args:
        path (path): Path to .csv file folder
        depth_max (int): Maximum distance to read the graph
        ac_operator (str): Arithmetic operator applied to the properties
        walk (str): Type of autocorrelation to be performed
    """
    keys = vectors[0].keys()

    with open(path/f'{ac_operator}_{walk}_d{depth_max}.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames = keys)
        dict_writer.writeheader()
        dict_writer.writerows(vectors)
        f.close()