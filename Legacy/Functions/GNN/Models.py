import torch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, ReLU, Tanh, Sigmoid

from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, Linear

class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(Linear(2 * in_channels, out_channels), Sigmoid(), Linear(out_channels, out_channels))
    
    def forward(self, sample):
        x, edge_index = sample.x, sample.edge_index
        return self.propagate(edge_index, x = x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return F.normalize(aggr_out)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim = 1)

class InvMassGNN(MessagePassing):

    def __init__(self, out_channels):
        super(InvMassGNN, self).__init__(aggr = "add")
        self.mlp = Seq(Linear(7, 64), ReLU(), Linear(64, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, out_channels))

    def forward(self, data):
        
        # Kinematics 
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        edge_index = data.edge_index
        x = torch.cat([e, pt, eta, phi], dim = 1)
        self.device = x.device
        return self.propagate(edge_index, x = x, dr = data.dr, m = data.m, dphi = data.dphi, edges = data.edge_index) 

    def message(self, x_i, x_j, dr, m, dphi):
        return self.mlp(torch.cat([x_i, dr, m, dphi], dim = 1))

    def update(self, aggr_out, edges):
        aggr_out = F.normalize(aggr_out)
        self.Adj_M = torch.zeros((len(aggr_out), len(aggr_out)), device = self.device)
        self.Adj_M[edges[0], edges[1]] = F.cosine_similarity(aggr_out[edges[0]], aggr_out[edges[1]])
        return aggr_out





class JetTaggingGNN(torch.nn.Module):

    def __init__(self,out_channels):
        super(JetTaggingGNN, self).__init__()

        self.conv1 = GCNConv(5, 1024)
        self.conv2 = GCNConv(1024, 3)
    
    def forward(self, data):
        e, pt, eta, phi, m = data.e, data.pt, data.eta, data.phi, data.m

        v = torch.cat([data.e, data.pt, data.eta, data.phi, data.m], dim = 1)
        v = self.conv1(v, data.edge_index)
        v = F.relu(v)
        v = self.conv2(v, data.edge_index)

        return v


