# Feature selection and ML
## Feature selection code and dense neural network and gaussian process models and associated training procedures.
### Feature selection
feature_selection_runs.py - contains code for an example run of the GBM model on the prediction task of the target barrier using the AABBA vector generated using the periodic (or generic) property set.
### Dense neural networks
dnn_runs.py - contains code specifying the training procedure for the DNN runs with different accumulated relevance input as calculated from the GBM. 
### Gaussian processes
gaussian_process_run.py - contains code specifying the training procedure for the gaussian processes, where you only need to specify accumulated relevance and type of input.
