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


class ExampleNet(torch.nn.Module):

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

class MLP(torch.nn.Module):
    """
    Equiwidth MLP of variable depth. The weights are Kaiming-He intialized.
    """
    def __init__(self, nfeatures, ntargets, depth, width):
        """
        nfeatures: int,
            number of features
        ntargets: int,
            number of targets
        depth: int,
            number of layers
        width: int,
            number of nodes in each layer
        """
        super(MLP, self).__init__()
        self.nfeatures = nfeatures
        self.ntargets = ntargets
        self.depth = depth
        self.layers = torch.nn.ModuleList([])
        self.relu = torch.nn.functional.relu
        first_layer = torch.nn.Linear(nfeatures, width)
        torch.nn.init.normal_(first_layer.weight, 0, np.sqrt(2.0/nfeatures))
        torch.nn.init.zeros_(first_layer.bias)
        self.layers.append(first_layer)
        if depth > 1:
            for i in range(1, depth):
                layer = torch.nn.Linear(width, width)
                torch.nn.init.normal_(layer.weight, 0, np.sqrt(2.0/width))
                torch.nn.init.zeros_(layer.bias)
                self.layers.append(layer)
            layer = torch.nn.Linear(width, ntargets)
            torch.nn.init.normal_(layer.weight, 0, np.sqrt(2.0/(width)))
            torch.nn.init.zeros_(layer.bias)
            self.final_layer = layer

        else:
            layer = torch.nn.Linear(width, ntargets)
            torch.nn.init.normal_(layer.weight, 0, np.sqrt(2.0/width))
            torch.nn.init.zeros_(layer.bias)
            self.final_layer = layer

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.relu(l(x))

        x = self.final_layer(x)
        return x

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

class DeepKernel(torch.nn.Module):
    """
    Deep Kernel
    """
    def __init__(self, nfeatures, nfeatures_out):
        """
        nfeatures: int,
            number of features
        ntargets: int,
            number of targets
        """
        super(DeepKernel, self).__init__()
        self.nfeatures = nfeatures
        self.ntargets = nfeatures_out
        self.layers = torch.nn.ModuleList([])
        self.relu = torch.nn.functional.relu
        input_layer = torch.nn.Linear(nfeatures, 128)
        torch.nn.init.normal_(input_layer.weight, 0, np.sqrt(2.0/nfeatures))
        torch.nn.init.zeros_(input_layer.bias)
        self.input_layer = input_layer
        hidden_layer = torch.nn.Linear(128, 128)
        torch.nn.init.normal_(hidden_layer.weight, 0, np.sqrt(2.0/128))
        torch.nn.init.zeros_(hidden_layer.bias)
        self.layer = hidden_layer
        output_layer = torch.nn.Linear(128, nfeatures_out)
        torch.nn.init.normal_(output_layer.weight, 0, np.sqrt(2.0/128))
        torch.nn.init.zeros_(output_layer.bias)
        self.output_layer = output_layer

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.layer(x))
        x = self.output_layer(x)
        return x


class DKLGP(gpytorch.models.ExactGP):
    """
    Deep Kernel Learning Gaussian Process
    """
    def __init__(self,
                 xtrain,
                 ytrain,
                 likelihood,
                 mean=gpytorch.means.ConstantMean(),
                 last_layer_dim = 3
                 ):
        super(DKLGP, self).__init__(xtrain, ytrain, likelihood)
        self.mean_module = mean
        rbf_kernel = gpytorch.kernels.RBFKernel(ard_dims=last_layer_dim)
        lin_kernel = gpytorch.kernels.LinearKernel()
        scale_lin_rbf = gpytorch.kernels.ScaleKernel(rbf_kernel*lin_kernel)
        """
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(ard_dims=last_layer_dim),
                num_dims=last_layer_dim,
                grid_size=100
                )
        )
        """
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel) #+ gpytorch.kernels.ScaleKernel(lin_kernel)
        nfeatures = xtrain.size(-1)
        self.deep_kernel = DeepKernel(nfeatures, last_layer_dim)
        # Scaling of features output from NN
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # Through the features extractor
        projected_x = self.deep_kernel(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_and_test(x_train, x_test, y_train, y_test, epochs, model, criterion, optimizer, scheduler):
    training_error = np.zeros(epochs)
    testing_error = np.zeros(epochs)
    r2_training = np.zeros(epochs)
    r2_testing = np.zeros(epochs)
    mae_training = np.zeros(epochs)
    mae_testing = np.zeros(epochs)
    count = 0
    iterator = tqdm.tqdm(range(epochs))
    best_loss = 100000
    for i in iterator:
        model.train()
        output = model(x_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_error[i] = loss.item()
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
            loss = criterion(preds, y_test)
            testing_error[i] = loss.item()
            r2_testing[i] = r2_score(preds.cpu(), y_test.cpu())
            mae_testing[i] = np.mean(np.abs(preds.cpu().numpy() - y_test.cpu().numpy()))
            train_preds = model(x_train)
            r2_training[i] = r2_score(train_preds.cpu(), y_train.cpu())
            mae_training[i] = np.mean(np.abs(train_preds.cpu().numpy() - y_train.cpu().numpy()))
            if loss < best_loss:
                best_model = model
                best_loss = loss.item()
        scheduler.step(np.mean(np.abs(preds.cpu().numpy() - y_test.cpu().numpy())))

    data = {"mse_train": training_error,
            "mse_test": testing_error,
            "r2_train": r2_training,
            "r2_test": r2_testing,
            "mae_train": mae_training,
            "mae_test": mae_testing}
    run_info = pd.DataFrame(data=data)

    return run_info, best_model
