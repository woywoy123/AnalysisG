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
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, Linear
from torch_geometric.data import Batch 

from PathNetOptimizer_cpp import PathCombination
from PathNetOptimizerCUDA_cpp import ToCartesianCUDA, PathMassCartesianCUDA


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

class PathNet(MessagePassing):
    def __init__(self, complex = 64, path = 64, hidden = 64, out = 50, Debug = False):
        super(PathNet, self).__init__("add")
        self.mlp_mass_complex = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, 1)) # DO NOT CHANGE OUTPUT DIM 
        self.mlp_comp = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, complex))
        self.mlp_path = Seq(Linear(-1, hidden), ReLU(), Linear(hidden, path))
        self.mlp = Seq(Linear(path + complex+2, path+complex), ReLU(), Linear(path+complex, hidden))
        self.N_Nodes = -1
        self.__Dyn_adj = -1
        self.__comb = []
        self.__cur = -1
        self.__PathMatrix = -1
        self.device = -1
        self.Debug = Debug
        if Debug:
            self.TMP = []

    def forward(self, data):

        def Performance(event):
            P = ToCartesianCUDA(event.eta, event.phi, event.pt, event.e)
            if type(event) == torch_geometric.data.data.Data:
                event = [event]
            else:
                event = event.to_data_list()

            if self.N_Nodes == -1:
                self.N_Nodes  = len(event[0].e)
                self.device = event[0].edge_index.device
            
            if self.__cur != len(event[0].e):
                self.__cur = len(event[0].e)

                Adj_Matrix = torch.zeros(self.__cur, self.__cur, device = self.device)
                Adj_Matrix[event[0].edge_index[0], event[0].edge_index[1]] = 1
                Combi = PathCombination(Adj_Matrix, self.__cur, 3)
                
                self.__Dyn_adj = torch.tensor([[i==j for i in range(self.N_Nodes)] for j in range(self.__cur)], dtype = torch.float, device = self.device)
                
                self.__comb = Combi[0]
                self.__PathMatrix = Combi[1]
                self.__PathMatrix = self.__PathMatrix[:, :].matmul(self.__Dyn_adj)
            
            torch.cuda.synchronize()
            n_events = len(event)
            adj_p, p_m = [], []
            for i in range(len(event)):
                s_ = i*self.__cur
                e_ = (i+1)*self.__cur

                m_cuda = PathMassCartesianCUDA(P[0][s_:e_], P[1][s_:e_], P[2][s_:e_], P[3][s_:e_], self.__comb)
                torch.cuda.synchronize()
                p_m.append(m_cuda)
                adj_p.append(self.__PathMatrix * m_cuda.reshape(self.__comb.shape[0], 1)[:, None])
            
            p_m = torch.cat(p_m, dim = 0)
            adj_p = torch.cat(adj_p, dim = 0)
            return adj_p, p_m, n_events, self.__comb.shape[0]

        # 1. ==== Assign the path mass to the adjacency matrix adj_p 
        edge_index = data.edge_index
        adj_p, path_m, n_event, pth_l = Performance(data)
        
        e_jxe_i_tmp = []
        comp_ej_tmp = []
        for i in range(n_event):
            blk = adj_p[i*pth_l:pth_l*(i+1)]

            # 2. ==== Project the sum of the topological mass states across complexity 
            e_jxe_i_tmp.append(torch.sum(blk, dim = 0))
            if self.Debug:
                self.TMP.append(torch.sum(blk, dim = 0))
       
            # 3. ==== Project along the complexity (the number of nodes included) and learn the complexity
            comp_ej_tmp.append(torch.sum(blk, dim = 1))
        
        # 2.1 ==== Learn the mass projection with dim (n*events * e_i) x e_j
        e_i = self.mlp_path(torch.cat(e_jxe_i_tmp, dim = 0))
       
        # 3.1 ==== Learn the different masses associated with path complexity (i.e. the number of nodes being included)
        comp_ej = self.mlp_mass_complex(torch.cat(comp_ej_tmp, dim = 0))
        
        # 3.2 ==== Set any connection with a mass to 1 and multiply (NOT DOT PRODUCT!!) with the learned mass value (-1 -> 1) to the matrix (n_events * e_i) x e_j
        adj_p = adj_p * comp_ej[:, None]
        
        # 4. ==== Split into n_events and project the learned mass values along the complexity axis to get a matrix e_i x e_j and sum the mass projection (non learned) to the learned mass complexity
        adj_p_tmp = []
        for i in range(n_event):
            adj_p_tmp.append(adj_p[i*pth_l:pth_l*(i+1)].sum(dim = 0))
            adj_p_tmp[i].add(e_jxe_i_tmp[i])
       
        # 4.1 ==== Learn this sum after concatinating the list to a matrix (n_events * e_i) x e_j
        adj_p_ei = self.mlp_comp(torch.cat(adj_p_tmp, dim = 0))


        return self.propagate(edge_index = edge_index, x = torch.cat([data.e, data.pt, data.eta, data.phi], dim = 1), edge_attr = torch.cat([e_i, adj_p_ei], dim = 1), dr = data.dr, dphi = data.dphi, edges = edge_index)

    def message(self, edge_index, x_i, x_j, dr, dphi, edge_attr):
        return self.mlp(torch.cat([edge_attr[edge_index[1].t()], dr, dphi], dim = 1))
    
    def update(self, aggr_out, edges): 
        aggr_out = F.normalize(aggr_out)
        self.Adj_M = torch.zeros((len(aggr_out), len(aggr_out)), device = self.device)
        self.Adj_M[edges[0], edges[1]] = F.cosine_similarity(aggr_out[edges[0]], aggr_out[edges[1]])
        return aggr_out




class PathNet_Old(MessagePassing):
    def __init__(self, PCut = 0.5, complex = 64, path = 64, hidden = 64, out = 20):
        super(PathNet, self).__init__("add")
        self.mlp_mass_complex = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, 1)) # DO NOT CHANGE OUTPUT DIM 
        self.mlp_comp = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, complex))
        self.mlp_path = Seq(Linear(-1, hidden), ReLU(), Linear(hidden, path))
        self.mlp = Seq(Linear(path + complex, path+complex), ReLU(), Linear(path+complex, hidden))
        self.PCut = PCut 
        self.N_Nodes = -1

    def forward(self, data):

        @njit(cache = True, parallel = True)
        def CreateLorentz(Lor_xyz, e, pt, eta, phi):
            
            for i in prange(Lor_xyz.shape[0]):
                Lor_xyz[i][0] = pt[i][0]*np.cos(phi[i][0])
                Lor_xyz[i][1] = pt[i][0]*np.sin(phi[i][0])
                Lor_xyz[i][2] = pt[i][0]*np.sinh(eta[i][0])
                Lor_xyz[i][3] = e[i][0]
        
        @njit(cache = True, parallel = True)
        def CalcPathMass(Lor_xyz, comb, path_m, adj_p):
            n = path_m.shape[0] 
            b = 0 
            inc = 0             
            for i in prange(n):
                
                if b == len(comb):
                    b = 0
                    inc += comb.shape[1]-1

                tmp = comb[b] + inc
                l = []
                for k in np.unique(tmp):
                    if k != -1 + inc:
                       l.append(k) 
                p = np.array(l) 
                v = np.zeros(4, dtype = np.float64) 
                for j in Lor_xyz[p]:
                    v[0] = v[0] + j[0]
                    v[1] = v[1] + j[1]
                    v[2] = v[2] + j[2]
                    v[3] = v[3] + j[3]
                path_m[i] = np.exp(0.5*np.log(v[3]*v[3] - v[2]*v[2] - v[1]*v[1] - v[0]*v[0])) / 1000
        
                for j in prange(len(l)-1):
                    adj_p[i, p[j] - inc, p[j+1] - inc] = path_m[i]
                    adj_p[i, p[j+1] - inc, p[j] - inc] = path_m[i]
                b += 1



        def Performance(e, pt, eta, phi, event):
            if type(event) == torch_geometric.data.data.Data:
                unique = torch.unique(event.edge_index).tolist()
                n_event = 1
            else:
                event = event.to_data_list()
                unique = torch.unique(event[0].edge_index).tolist()
                n_event = len(event)
           
            if self.N_Nodes == -1:
                self.N_Nodes = len(unique)

            e = np.array(e.tolist())
            pt = np.array(pt.tolist())
            eta = np.array(eta.tolist())
            phi = np.array(phi.tolist())
        
            Lor_xyz = np.zeros((len(e), 4))
            CreateLorentz(Lor_xyz, e, pt, eta, phi) 
            
            tmp = [] 
            l = len(unique)
            for i in range(1, l):
                p = list(combinations(unique, r = i+1))
                p = [list(k) for k in p]
                for k in p:
                    k += [k[0]]
                    k += [-1]*(self.N_Nodes - len(k)+1)
                tmp += p
            
            path = np.array(tmp)
            p_m = np.zeros(path.shape[0]*n_event, dtype = np.float32)
            adj_p = np.zeros((path.shape[0]*n_event, self.N_Nodes, self.N_Nodes), dtype = float)
            CalcPathMass(Lor_xyz, path, p_m, adj_p)
            return adj_p, p_m, n_event, path.shape[0]

        # 1. ==== Assign the path mass to the adjacency matrix adj_p 
        edge_index = data.edge_index
        adj_p, path_m, n_event, pth_l = Performance(data.e, data.pt, data.eta, data.phi, data)
        block = torch.tensor(adj_p, device = edge_index.device).float()

        e_jxe_i_tmp = []
        comp_ej_tmp = []
        for i in range(n_event):
            blk = block[i*pth_l:pth_l*(i+1)]

            # 2. ==== Project the sum of the topological mass states across complexity 
            e_jxe_i_tmp.append(torch.sum(blk, dim = 0))
       
            # 3. ==== Project along the complexity (the number of nodes included) and learn the complexity
            comp_ej_tmp.append(torch.sum(blk, dim = 1))
        
        # 2.1 ==== Learn the mass projection with dim (n*events * e_i) x e_j
        e_i = self.mlp_path(torch.cat(e_jxe_i_tmp, dim = 0))
       
        # 3.1 ==== Learn the different masses associated with path complexity (i.e. the number of nodes being included)
        comp_ej = self.mlp_mass_complex(torch.cat(comp_ej_tmp, dim = 0))
        
        # 3.2 ==== Set any connection with a mass to 1 and multiply (NOT DOT PRODUCT!!) with the learned mass value (-1 -> 1) to the matrix (n_events * e_i) x e_j
        adj_p[adj_p > 0] = 1 
        adj_p = torch.tensor(adj_p, device = edge_index.device).float()
        adj_p = adj_p * comp_ej[:, None]
        
        # 4. ==== Split into n_events and project the learned mass values along the complexity axis to get a matrix e_i x e_j and sum the mass projection (non learned) to the learned mass complexity
        adj_p_tmp = []
        for i in range(n_event):
            adj_p_tmp.append(adj_p[i*pth_l:pth_l*(i+1)].sum(dim = 0))
            adj_p_tmp[i].add(e_jxe_i_tmp[i])
       
        # 4.1 ==== Learn this sum after concatinating the list to a matrix (n_events * e_i) x e_j
        adj_p_ei = self.mlp_comp(torch.cat(adj_p_tmp, dim = 0))

        return self.propagate(edge_index = edge_index, x = torch.cat([data.e], dim = 1), edge_attr = torch.cat([e_i, adj_p_ei], dim = 1))

    def message(self, edge_index, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([edge_attr[edge_index[1].t()]], dim = 1))
    
    def update(self, aggr_out): 
        aggr_out = F.normalize(aggr_out)
        adj = torch.sigmoid(aggr_out.matmul(aggr_out.t()))
        l = adj.shape[0]

        ones = torch.zeros(adj.shape, device = adj.device)
        for i in range(l):
            ones[l - i-1] = torch.tensor([[1]*(l-i) + [0]*i])

        adj[adj <= self.PCut] = 0
        adj_ = adj.matmul(ones)
        adj_i = adj_.sum(dim = 0)
        adj_j = adj_.sum(dim = 1)
        
        adj_sum = adj_i + adj_j.flip(dims = [0])
        step = (adj_sum.max() - adj_sum.min())/len(adj_sum)


        c_ = np.array(torch.round(adj_sum).tolist()).reshape(-1, 1)
        clu = list(AgglomerativeClustering(n_clusters = None, distance_threshold = float(step)).fit(c_).labels_)

        self.n_cluster = len(list(set(clu)))
        self.Adj_M = adj
        self.cluster = clu
        return aggr_out


