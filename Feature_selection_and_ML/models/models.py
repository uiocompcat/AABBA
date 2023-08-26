import math
import torch
from torch import nn
import gpytorch
import numpy as np
import pandas as pd
from numpy.random import default_rng
import tqdm
from collections import OrderedDict
from sklearn.metrics import r2_score
torch.set_default_dtype(torch.float)


class DNN(torch.nn.Module):

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, hidden_layers: int):
        super(ExampleNet, self).__init__()
        layers_ordered_dict = OrderedDict()
        layers_ordered_dict["input"] = nn.Linear(input_nodes, hidden_nodes)
        layers_ordered_dict["ReLU_1"] = nn.ReLU()
        for i in range(hidden_layers):
            layer_name = "hidden_" + str(i)
            relu_name = "ReLU_" + str(i)
            layers_ordered_dict[layer_name] = nn.Linear(hidden_nodes, hidden_nodes)
            layers_ordered_dict[relu_name] = nn.ReLU()
        layers_ordered_dict["output"] = nn.Linear(hidden_nodes, output_nodes)

        self.layers = nn.Sequential(
            layers_ordered_dict
        )

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x.view(-1)


class GP(gpytorch.models.ExactGP):
    def __init__(self, xtrain, ytrain, likelihood):
        super(GP, self).__init__(xtrain, ytrain, likelihood)
        # RBF kernel
        rbf_kernel = gpytorch.kernels.RBFKernel(ard_dims=xtrain.shape[-1])
        # Linear kernel
        lin_kernel = gpytorch.kernels.LinearKernel()

        # Define linear RBF kernel
        scale_lin_rbf = gpytorch.kernels.ScaleKernel(rbf_kernel*lin_kernel)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = scale_lin_rbf
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
