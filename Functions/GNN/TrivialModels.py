import torch
from torch.nn import Sequential as Seq, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from torch import nn

from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, Linear


class GraphNN(nn.Module):
    
    def __init__(self, inputs = 1):
        super(GraphNN, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(1, 64), 
                nn.ReLU(), 
                nn.Linear(64, 32), 
                nn.ReLU(), 
                nn.Linear(32, 1)
        )
        
        self.L_Signal = "CEL"
        self.C_Signal = True
        self.O_Signal = 0

    def forward(self, G_Signal, edge_index):
        self.O_Signal = self.layers(G_Signal.view(-1, 1))

        return self.O_Signal


class NodeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(
                Linear(2 * in_channels, 2), 
                ReLU(), 
                Linear(2, out_channels)
        )
    
        self.L_x = "CEL"
        self.C_x = True
        self.O_x = 0
    
    def forward(self, N_x, N_Sig, edge_index):
        x = torch.cat((N_x, N_Sig), dim = 1)
        self.O_x = self.propagate(edge_index, x = x)        
        return self.O_x

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return F.normalize(aggr_out)


class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "add") 
        self.mlp = Seq(
                Linear(in_channels, 256),
                Linear(256, out_channels)
        )
    
        self.L_x = "MSEL"
        self.N_x = False
        self.C_x = False

        self.O_x = 0
    
    def forward(self, E_x, edge_index):
        self.__edge = E_x
        self.O_x = self.message()
        return self.O_x

    def message(self):
        tmp = self.__edge.view(-1, 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return F.normalize(aggr_out)
