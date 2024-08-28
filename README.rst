========================
AABBA 
========================

.. project-description-start

AABBA is a Graph Kernel Python library for applying autocorrelation (AC) functions.
It transforms molecular graphs into a fixed-length vector that combines generic properties (GP) and 
natural bond orbital (NBO) properties. 

*NB! Generic properties (GP) and periodic-table (PT) features are employed indistinctly.*

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

Define the AABBA parameters in the first lines of **ac_NBO_multithread.py** and **ac_PT_multithread.py** files:

        1) Select the parameters to perform the autocorrelation functions accordingly:

        - **ac_operator**: origin of the autocorrelation (M or F) and arithmetic operator (A, D, S, R) applied to the properties. 

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
        
        *NB! MC refers to metal-centered autocorrelation and F refers to full autocorrelation.*

        - **walk**: types of autocorrelation. The different modes to read the chemical graph.

        +---------------+---------------------------------------------------------------------------------------------------+
        | walk variable  (str)                                                                                              |
        +===============+===================================================================================================+
        | AA            | atom-atom correlation                                                                             |
        +---------------+---------------------------------------------------------------------------------------------------+
        | BBavg         | bond-bond autocorrelation with averaging-superbond (only for MC)                                  |
        +---------------+---------------------------------------------------------------------------------------------------+
        | BB            | bond-bond autocorrelation with summing-superbond (for MC), and with full bond-bond (F)            |
        +---------------+---------------------------------------------------------------------------------------------------+
        | AB            | bond-atom autocorrelation                                                                         |
        +---------------+---------------------------------------------------------------------------------------------------+
        | ABBAavg       | implicit autocorrelation with averaging-superbond (only for MC); select the model 1, 2, 3, 4, 5   |
        +---------------+---------------------------------------------------------------------------------------------------+
        | ABBA          | implicit autocorrelation with individual bond (only for F); select the model 1, 2, 3, 4, 5        |
        +---------------+---------------------------------------------------------------------------------------------------+

        *NB! According to the article:*

            *AABBA(I) = AA ⊕ BBavg ⊕ AB, therefore, it is necessary to obtain them separately, (i.e. first AA, then BBavg, and AB) and concatenate them afterwards.* 

            *AABBA(II) is obtained using AABBAavg and ABBA in the code. We also need to indicate the model number to obtain each of the different five possibilities*

        - **model_number**: 1, 2, 3 to be performed with periodic table features (PT); and 4, 5 to be performed with nbo features (NBO). The attributes contain the following features:
        
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

        - **depth_max** (int): maximum depth of the autocorrelation function.


The graphs (.gml files) are stored in the folllowing directories:

- 'PT_graphs' for generic property graphs.

- 'uNatQ_graphs' for NBO property graphs.

If the location of these folders is different, adjust the path_to_gml variable in your script accordingly.

Resulting Vectors: The vectors generated from processing the graphs are saved in the 'vectors_ABBA' folder.


Code Configuration and Customization
------------------------------------
This code is designed to work with graphs extracted from https://github.com/uiocompcat/HyDGL. If you have graphs with different properties or need to customize certain aspects, follow these steps:

- Adjust Graph Properties:
File: **graph_info.py**. Define and adjust the properties and features of your graphs according to your requirements.

- Customize Decimal Precision:
File: **utilities.py**. Modify how decimals are rounded to meet your specific needs.

Ensure that these customizations align with the rest of your code to maintain compatibility and accuracy.

For more information, please refer to the preprint: 

Citation 
--------
Morán-González L, Betten JE, Kneiding H, Balcells D. AABBA: Atom–Atom Bond–Bond Bond–Atom Graph Kernel for Machine Learning on Molecules and Materials. ChemRxiv. 2023; doi:10.26434/chemrxiv-2023-5wbkr

Contact 
-------
l.m.gonzalez@smn.uio.no


