from torch_geometric.nn import MessagePassing, GCNConv, TopKPooling
from torch_geometric.data import Batch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from skhep.math.vectors import LorentzVector
from torch_scatter import scatter
import torch 
import math

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


def Mass(data, labels, classification):
    e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
    L = [LorentzVector() for i in range(classification)]
    
    inx = []
    for i in range(len(labels)):
        t = LorentzVector()
        t.setptetaphie(pt[i], eta[i], phi[i], e[i])
        L[int(labels[i])] += t
        inx.append(int(labels[i]))
    L = [[i.mass] for i in L]
    return torch.tensor(L), inx

class InvMassGNN(MessagePassing):

    def __init__(self, out_channels):
        super(InvMassGNN, self).__init__(aggr = "add")
        self.mlp = Seq(Linear(6, 6), ReLU(), Linear(6, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, out_channels))
    
    def forward(self, data):

        # Kinematics 
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        edge_index = data.edge_index
        x = torch.cat([e, pt, eta, phi], dim = 1)

        return self.propagate(edge_index, x = x, dr = data.dr, m = data.m) 

    def message(self, x_i, x_j, dr, m):
        return self.mlp(torch.cat([x_i, dr, m], dim = 1))

    def update(self, aggr_out):
        return F.normalize(aggr_out)
        


class InvMassAggr(MessagePassing):

    def __init__(self, out_channels):
        super(InvMassAggr, self).__init__(aggr = "add")
        self.mlp = Seq(Linear(6, 6), ReLU(), Linear(6, 12), ReLU(), Linear(12, 12), ReLU(), Linear(12, out_channels))
    
    def forward(self, data):

        # Kinematics 
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        d_r, edge_index = data.d_r, data.edge_index
        x = torch.cat([e, pt, eta, phi], dim = 1)
        return self.propagate(edge_index, x = x) 

    def message(self, x_i, x_j):
        dr = []
        m = []
        de = []
        for i, j in zip(x_i, x_j):
            del_R = math.sqrt(math.pow(i[2] - j[2], 2) + math.pow(i[3] - j[3], 2))
            dr.append([del_R])
            T_i = LorentzVector()
            T_i.setptetaphie(i[1], i[2], i[3], i[0])

            T_j = LorentzVector()
            T_j.setptetaphie(j[1], j[2], j[3], j[0])
            T = T_j + T_i
            m.append([T.mass])
        
        dr = torch.tensor(dr)
        m = torch.tensor(m)
        dr = torch.cat([x_i, dr, m], dim = 1)
        return self.mlp(dr)
    
    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim = self.node_dim, dim_size = dim_size, reduce = "add")

    def update(self, aggr_out):
        return F.normalize(aggr_out)





