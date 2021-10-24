from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
import torch 

class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(Linear(2 * in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x = x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        F.normalize(aggr_out)
        return aggr_out


