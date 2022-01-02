
import math
from itertools import combinations

import numba
from numba import njit, prange

import numpy as np
from skhep.math.vectors import LorentzVector
from sklearn.cluster import AgglomerativeClustering

import torch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, ReLU, Tanh, Sigmoid

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, GCNConv, Linear
from torch_geometric.data import Batch 

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
        self.mlp = Seq(Linear(6, 32), ReLU(), Linear(32, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, out_channels))
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
        self.mlp = Seq(Linear(8 +2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 63))
        self.Mmlp = Seq(Linear(63 + 1, 63 + 1), ReLU(), Linear(63+1, out_channels))
    def forward(self, data):

        # Kinematics 
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        x = torch.cat([data.e, data.pt, data.eta, data.phi], dim = 1)
        return self.propagate(edge_index = data.edge_index, x = x, dr = data.dr, m = data.m, y = data.y) 

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
        dr = torch.cat([x_i, dr, m, abs(x_i - x_j)], dim = 1)
        return self.mlp(dr)
    
    def aggregate(self, inputs, index, dim_size):
        #https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        # - Basically a form of hashing where the index is used to extract values from the inputs and sums these values @ index value
        return scatter(inputs, index, dim = self.node_dim, dim_size = dim_size, reduce = "add")

    def update(self, aggr_out, x, edge_index, y):
        #aggr_out = F.normalize(aggr_out)
        #edge_weight = torch.sigmoid(torch.matmul(aggr_out, aggr_out.t()))

        
        _, p = aggr_out.max(1)
        M = [LorentzVector() for i in range(len(x))]
        for e_i, e_j in zip(edge_index[0], edge_index[1]):
            # incoming kinematic quantities  x_j -> x_i            
            #if edge_weight[e_j][e_i] > self.Threshold:
            
            if p[e_i] == p[e_j]:
                v = LorentzVector()
                v.setptetaphie(x[e_j][1], x[e_j][2], x[e_j][3], x[e_j][0])
                M[e_i]+=v
        
        # Include own mass in the calculation
        for e_i in set(edge_index[0].tolist()):
            v = LorentzVector()
            v.setptetaphie(x[e_i][1], x[e_i][2], x[e_i][3], x[e_i][0])
            M[e_i] += v

        M = torch.cat([aggr_out, torch.tensor([[i.mass] for i in M])], dim = 1)
        return F.normalize(self.Mmlp(F.normalize(M)))


class PathNet(MessagePassing):
    def __init__(self, PCut = 0.5, complex = 64, path = 64, hidden = 64, out = 20):
        super(PathNet, self).__init__("add")
        self.mlp_mass_complex = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, 1)) # DO NOT CHANGE OUTPUT DIM 
        self.mlp_comp = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, complex))
        self.mlp_path = Seq(Linear(-1, hidden), ReLU(), Linear(hidden, path))
        self.mlp = Seq(Linear(path + complex, path+complex), ReLU(), Linear(path+complex, out))
        self.PCut = PCut 
        self.Debug = False

    def forward(self, data):
        
        def PathMass(Lor, com):
            outp = []
            for i in com:
                v = LorentzVector()
                for k in i:
                    v += Lor[k]
                outp.append(v.mass/1000)
            return outp

        @njit(cache = True, parallel = True)
        def MakeMatrix(out, path_m, path):
            n = path.shape[0]
            for x in prange(n):
                m = path_m[x]
                tmp = path[x]
                pth = []
                for k in range(len(tmp)):
                    if tmp[k] != -1:
                        pth.append(tmp[k])

                m = m/len(pth) 
                for j in range(len(pth)-1):
                    out[x, pth[j], pth[j+1]] = m
                    out[x, pth[j+1], pth[j]] = m
               
                e_s = pth[0]
                e_e = pth[len(pth)-1]
                if out[x, e_s, e_e] == 0:
                    out[x, e_s, e_e] = m
                    out[x, e_e, e_s] = m

        edge_index = data.edge_index
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        unique = np.unique(edge_index.tolist())
        l = len(unique)
        
        Lor = []
        for i in unique:
            v = LorentzVector()
            v.setptetaphie(pt[i], eta[i], phi[i], e[i])
            Lor.append(v)
  
        path = []
        path_m = []
        for i in range(1,l):
            p = list(combinations(unique, r = i+1))
            path_m += PathMass(Lor, p)
            p = [list(k) for k in p]
            for k in p:
                k += [k[0]]
                k += [-1]*(l - len(k)+1)
            path += p
        
        if self.Debug:
            self.Path = path
            self.Path_M = path_m

        # 1. ==== Assign the path mass to the adjacency matrix adj_p 
        adj_p = np.zeros((len(path_m), l, l), dtype = float)
        path_m = np.array(path_m, dtype = float)
        path = np.array(path, dtype = int)
        MakeMatrix(adj_p, path_m, path)
        block = torch.tensor(adj_p, device = edge_index.device).float()
        
        # 2. ==== Project the sum of the topological mass states across complexity 
        e_jxe_i = torch.sum(block, dim = 0)
        e_i = self.mlp_path(e_jxe_i)

        # 3. ==== Project along the complexity and learn the complexity
        comp_ej = torch.sum(block, dim = 1)
        comp_ej = self.mlp_mass_complex(comp_ej)
        adj_p[adj_p > 0] = 1     
        adj_p = torch.tensor(adj_p, device = edge_index.device).float()
        adj_p = adj_p * comp_ej[:, None]
        adj_p = adj_p.sum(dim = 0)
        adj_p = adj_p.add(e_jxe_i)
        adj_p_ei = self.mlp_comp(adj_p)

        return self.propagate(edge_index = edge_index, x = torch.cat([e], dim = 1), edge_attr = torch.cat([e_i, adj_p_ei], dim = 1))

    def message(self, edge_index, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([edge_attr[edge_index[1].t()]], dim = 1))
    
    def update(self, aggr_out): 
        aggr_out = F.normalize(aggr_out)
        #adj = torch.sigmoid(aggr_out.matmul(aggr_out.t()))
        #l = adj.shape[0]

        #ones = torch.zeros(adj.shape, device = adj.device)
        #for i in range(l):
        #    ones[l - i-1] = torch.tensor([[1]*(l-i) + [0]*i])

        #adj[adj <= self.PCut] = 0
        #adj_ = adj.matmul(ones)
        #adj_i = adj_.sum(dim = 0)
        #adj_j = adj_.sum(dim = 1)
        #
        #adj_sum = adj_i + adj_j.flip(dims = [0])
        #step = (adj_sum.max() - adj_sum.min())/len(adj_sum)


        #c_ = np.array(torch.round(adj_sum).tolist()).reshape(-1, 1)
        #clu = list(AgglomerativeClustering(n_clusters = None, distance_threshold = float(step)).fit(c_).labels_)

        #self.NCluster = len(list(set(clu)))
        #self.Adj_M = adj
        #self.Cluster = clu
        return aggr_out
