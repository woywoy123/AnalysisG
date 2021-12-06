from torch_geometric.nn import MessagePassing, GCNConv, TopKPooling
from torch_geometric.data import Batch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from skhep.math.vectors import LorentzVector
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
    v = data.x
    L = [LorentzVector() for i in range(classification)]
    
    inx = []
    for n, x_i in zip(labels, v):
        t = LorentzVector()
        t.setptetaphie(x_i[2], x_i[1], x_i[3], x_i[0])
        L[int(n)] += t
        inx.append(int(n))
    L = [[i.mass] for i in L]
    return torch.tensor(L), inx

class InvMassGNN(MessagePassing):

    def __init__(self, out_channels):
        super(InvMassGNN, self).__init__(aggr = "add")
        self.mlp = Seq(Linear(6, 4), Sigmoid(), Linear(4, out_channels))       
    
    def forward(self, data):
        # Kinematics 
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        d_r, edge_index = data.d_r, data.edge_index
        y = data.y
        y = y.t().contiguous().squeeze()
        #print(Mass(data, y, 4)) 
        
        x = torch.cat([e, pt, eta, phi], dim = 1)
        return self.propagate(edge_index, x = x) 

    def message(self, x_i, x_j):
        dr = []
        m = []
        for i, j in zip(x_i, x_j):
            dr.append([math.sqrt(math.pow(i[2] - j[2], 2) + math.pow(i[3] - j[3], 2))])
            T_i = LorentzVector()
            T_i.setptetaphie(i[1], i[2], i[3], i[0])

            T_j = LorentzVector()
            T_i.setptetaphie(i[1], i[2], i[3], i[0])
            T = T_j + T_i
            m.append([T.mass])

        dr = torch.tensor(dr)
        m = torch.tensor(m)
        dr = torch.cat([x_i, dr, m], dim = 1)
        return self.mlp(dr)

    def update(self, aggr_out):
        print(aggr_out)
        return F.normalize(aggr_out)
        








