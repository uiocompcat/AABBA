import torch.nn as nn
import torch.nn.functional as F

class ExampleNet(nn.Module):

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        super(ExampleNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_nodes, hidden_nodes),
            nn.ReLU(),
            #nn.Linear(hidden_nodes, hidden_nodes), # added new linear + relu
            #nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, output_nodes)
        )

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x.view(-1)
