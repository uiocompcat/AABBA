========================
ABBA 
========================

.. project-description-start

ABBA is a Python library for applying autocorrelation (AC) functions. It
provides a fixed-length vector which combines generic properties (GP) and 
natural bond orbital (NBO) properties. 

**Generic properties (GP) and periodic-table (PT) features are employed indistinctly.

.. project-description-end

Requirements
------------
* Python 3.7.0+
* NumPy 1.21.5+
* NetworkX 2.6.3

Scripts
-------
* graph_info.py - read the chemical graph and extract the indexes of the nodes and edges at different depths. It also provides the labels of the features.
* ac_functions.py - perform the autocorrelation functions.
* utilities - tools for manipulating the data.
* ac_PT_multithread - parallel implementation to perform the autocorrelation functions with periodic-table features.
* ac_NBO_multithread - parallel implementation to perform the autocorrelation functions with nbo features.

How to use
----------
Clone the code.
    
        git clone

Install the requirements.

Manipulate **ac_NBO_multithread.py** and **ac_PT_multithread.py** files:

        1) Select the parameters to perform the autocorrelation functions.

        - ac_operator: arithmetic operator applied to the properties. 

        +----------+-------------------------------------------+
        | ac_operator (str)                                    |
        +=========+============================================+
        | MA      | metal-centered autocorrelation             |
        +---------+--------------------------------------------+
        | MD      | metal-centered deltametric autocorrelation |
        +---------+--------------------------------------------+
        | MS      | metal-centered summetric autocorrelation   |
        +---------+--------------------------------------------+
        | MR      | metal-centered ratiometric autocorrelation |
        +---------+--------------------------------------------+
        | FA      | full autocorrelation                       |
        +---------+--------------------------------------------+
        | FD      | full deltametric autocorrelation           |
        +---------+--------------------------------------------+
        | FS      | full summetric autocorrelation             |
        +---------+--------------------------------------------+
        | FR      | full ratiometric autocorrelation           |
        +---------+--------------------------------------------+
        
        **MC refers to metal-centered autocorrelation and F refers to full autocorrelation

        - walk: types of autocorrelation. The different modes to read the chemical graph.

        +---------------+------------------------------------------------------------------------------+
        | walk variable  (str)                                                                         |
        +===============+==============================================================================+
        | AA            | atom-atom correlation                                                        |
        +---------------+------------------------------------------------------------------------------+
        | BBavg         | bond-bond autocorrelation with bond average (only MC)                        |
        +---------------+------------------------------------------------------------------------------+
        | BB            | bond-bond autocorrelation with superbond (for MC), and full bond-bond (F)    |
        +---------------+------------------------------------------------------------------------------+
        | AB            | bond-atom autocorrelation                                                    |
        +---------------+------------------------------------------------------------------------------+
        | nBB           | new bond-bond autocorrelation (only for PT features)                         |
        +---------------+------------------------------------------------------------------------------+

        - model_number: to be performed only with periodic table features (PT). The attributes of the nodes contain the following features:
        
        +----------+---------------------------------------+
        | model_number (str)                               |
        +========+=========================================+
        | 1      | Zi, Zj, Ti, Tj, Xi, Xj, d, BO, I        |
        +--------+-----------------------------------------+
        | 2      | Zi, Zj, Ti, Tj, Xi-Xj, d, BO, I         |
        +--------+-----------------------------------------+
        | 3      | Zi, Zj, Ti, Tj, Xi-Xj, Si, Sj, BO, I    |
        +--------+-----------------------------------------+

        (i and j are the nodes of the edges)

        - depth_max (int): maximum depth of the autocorrelation function.

|

        2) Select the autocorrelation function to be performed and its return vector. Comment the rest of the functions.

|

        3) Select the feature type (feature_type)

|

        4) Adjust the path to the folder that contains the .gml files (path_to_gml).

|

        5) Create a folder to save the results (path_to_folder).






