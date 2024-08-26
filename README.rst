========================
AABBA 
========================

.. project-description-start

AABBA is a Graph Kernel Python library for applying autocorrelation (AC) functions..
It transforms molecular graphs into a fixed-length vector that combines generic properties (GP) and 
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
* utilities.py - tools for manipulating the data.
* ac_PT_multithread.py - parallel implementation to perform the autocorrelation functions with periodic-table features.
* ac_NBO_multithread.py - parallel implementation to perform the autocorrelation functions with nbo features.

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
        | MA      | metal-centered product autocorrelation     |
        +---------+--------------------------------------------+
        | MD      | metal-centered deltametric autocorrelation |
        +---------+--------------------------------------------+
        | MS      | metal-centered summetric autocorrelation   |
        +---------+--------------------------------------------+
        | MR      | metal-centered ratiometric autocorrelation |
        +---------+--------------------------------------------+
        | FA      | full product autocorrelation               |
        +---------+--------------------------------------------+
        | FD      | full deltametric autocorrelation           |
        +---------+--------------------------------------------+
        | FS      | full summetric autocorrelation             |
        +---------+--------------------------------------------+
        | FR      | full ratiometric autocorrelation           |
        +---------+--------------------------------------------+
        
        **MC refers to metal-centered autocorrelation and F refers to full autocorrelation

        - walk: types of autocorrelation. The different modes to read the chemical graph.

        +---------------+----------------------------------------------------------------------------------------+
        | walk variable  (str)                                                                                   |
        +===============+========================================================================================+
        | AA            | atom-atom correlation                                                                  |
        +---------------+----------------------------------------------------------------------------------------+
        | BBavg         | bond-bond autocorrelation with bond average (only MC)                                  |
        +---------------+----------------------------------------------------------------------------------------+
        | BB            | bond-bond autocorrelation with superbond (for MC), and with full bond-bond (F)         |
        +---------------+----------------------------------------------------------------------------------------+
        | AB            | bond-atom autocorrelation                                                              |
        +---------------+----------------------------------------------------------------------------------------+
        | ABBAavg       | implicit autocorrelation with bond average (only MC); select the model 1, 2, 3, 4, 5   |
        +---------------+----------------------------------------------------------------------------------------+
        | ABBA          | implicit autocorrelation with individual bond (only F); select the model 1, 2, 3, 4, 5 |
        +---------------+----------------------------------------------------------------------------------------+

        **According to the article: 

            AABBA(I) = AA ⊕ BBavg ⊕ AB, therefore, it is necessary to obtain them separately, (i.e. first AA, then BBavg, and AB) and concatenate them afterwards. // 
            AABBA(II) is obtained using AABBAavg and ABBA in the code. We also need to indicate the model number to obtain each of the different five possibilities. 

        - model_number: 1, 2, 3 to be performed with periodic table features (PT); and 4, 5 to be performed with nbo features (NBO). The attributes contain the following features:
        
        +----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | model_number (str)                                                                                                                                                          |        
        +========+====================================================================================================================================================================+
        | 1      | Z\ :sub:`i`, Z\ :sub:`j`, T\ :sub:`i`, T\ :sub:`j`, X\ :sub:`i`, X\ :sub:`j`, d, BO, I                                                                             |
        +--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | 2      | Z\ :sub:`i`, Z\ :sub:`j`, T\ :sub:`i`, T\ :sub:`j`, X\ :sub:`i`-X\ :sub:`j`, d, BO, I                                                                              |
        +--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | 3      | Z\ :sub:`i`, Z\ :sub:`j`, T\ :sub:`i`, T\ :sub:`j`, X\ :sub:`i`-X\ :sub:`j`, S\ :sub:`i`, S\ :sub:`j`, BO, I                                                       |
        +--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | 4      | qNat\ :sub:`i`, qNat\ :sub:`j`, VNat\ :sub:`i`, VNat\ :sub:`j`, Ns\ :sub:`i`, Ns\ :sub:`j`, Np\ :sub:`i`, Np\ :sub:`j`, Nd\ :sub:`i`, Nd\ :sub:`j`, NLP\ :sub:`i`, |
        |        | NLP\ :sub:`j`, NLV\ :sub:`i`, NLV\ :sub:`j`, BD, BONat, NBN , BNs, BNp, BNd, NBN∗ , BN∗s , BN∗p , BN∗d , I                                                         |
        +--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | 5      | qNat\ :sub:`i`, qNat\ :sub:`j`, VNat\ :sub:`i`, VNat\ :sub:`j`, NLP\ :sub:`i`, NLP\ :sub:`j`, LPE\ :sub:`i`, LPE\ :sub:`j`, LP∆E\ :sub:`i`, LP∆E\ :sub:`j`,        |
        |        | NLV\ :sub:`i`, NLV\ :sub:`j`, LVE\ :sub:`i`, LVE\ :sub:`j`, LV∆E\ :sub:`i`, LV∆E\ :sub:`j`, BD, BONat, NBN , BNE , BN∆E , NBN∗ , BN∗E , BN∗∆E , I                  |
        +--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+


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






