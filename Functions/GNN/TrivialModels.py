import torch
import torch.nn.functional as F
from torch import nn


class GraphNN(nn.Module):
    
    def __init__(self, inputs = 1):
        super(GraphNN, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(1, 64), 
                nn.ReLU(), 
                nn.Linear(64, 32), 
                nn.ReLU(), 
                nn.Linear(32, 2)
        )
        self.L_Signal = "CEL"
        self.C_Signal = True

    def forward(self, data):
        self.G_Signal = self.layers(data.G_Signal.view(-1, 1))

