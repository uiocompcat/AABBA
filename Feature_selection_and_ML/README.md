# Feature selection and ML
## Feature selection code and dense neural network and gaussian process models and associated training procedures.
### Example application using the AABBA autocorrelation vectors
This is an example where the dimensionalities of large AABBA autocorrelation vectors were reduced with respect to relevance calculated from Gradient Boosting Machines (GBMs). The resulting reduced autocorrelation vectors were used as inputs to dense neural networks and gaussian processes, demonstrating the effectiveness of this approach. 
### Feature selection
feature_selection_runs.py - contains code for an example run of the GBM model on the prediction task of the target barrier using the AABBA vector generated using the periodic (or generic) property set.
### Dense neural networks
dnn_runs.py - contains code specifying the training procedure for the DNN runs with different accumulated relevance input as calculated from the GBM. 
### Gaussian processes
gaussian_process_run.py - contains code specifying the training procedure for the gaussian processes, where you only need to specify accumulated relevance and type of input.
