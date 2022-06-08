import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

# This model is a replica of the Paper: "Semi-supervised Classification with Graph Convolutional Networks" ~ https://arxiv.org/abs/1609.02907
class BaseLineModel(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "max")
       
        self._mlp = Seq(Linear(2*in_channels, out_channels), 
                        ReLU(), 
                        Linear(out_channels, out_channels))

        self.O_Index = None
        self.L_Index = "CEL"
        self.C_Index = True
        self.N_Index = True

        self.Device = ""

    
    def forward(self, edge_index, N_Index):
        self.O_Index = self.propagate(edge_index, x = N_Index) 
        return self.O_Index
    
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self._mlp(tmp)


import LorentzVector as LV
from torch_geometric.utils import *
class BaseLineModelAdvanced(MessagePassing):

    def __init__(self):
        super().__init__(aggr = "max")   

        self.O_Signal = None
        self.L_Signal = "CEL"
        self.C_Signal = True
        self.N_Signal = False

        self._mlp = Seq(Linear(11, 256), 
                        ReLU(),
                        Linear(256, 1024),
                        ReLU(),
                        Linear(1024, 12))

        self._mlp_dr = Seq(Linear(1, 256), 
                        ReLU(),
                        Linear(256, 1024),
                        ReLU(),
                        Linear(1024, 12))


        self._mlp_m = Seq(Linear(1, 1024), 
                          ReLU(),
                          Linear(1024, 1024),
                          ReLU(),
                          Linear(1024, 2))

        self._mlp_edge = Seq(Linear(12 + 12 + 2, 256),
                            ReLU(),
                            Linear(256, 2))

        self._Pmu = None
        self._iter = 0
        self._edge_index = None

    def forward(self, edge_index, N_pT, N_eta, N_phi, N_energy):
        if self._iter == 0:
            self._P_mu = LV.TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
            self._edge_index = edge_index
        
        if self._iter > 0 or len(edge_index[0]) == 0:
            self._iter = 0
            return self.O_Signal

        prop = self.propagate(edge_index, x = self._P_mu)
        mass = self._mlp_m(LV.MassFromPxPyPzE(self._P_mu[edge_index[0]] + self._P_mu[edge_index[1]]))
        dr = self.deltaR(edge_index, N_eta, N_phi)

        if self._iter == 0:
            self.O_Signal = self._mlp_edge(torch.cat([prop[edge_index[0]], mass, dr], dim = 1))
        else: 
            self.O_Signal[edge_index[0]] += self._mlp_edge(torch.cat([prop[edge_index[0]], mass, dr], dim = 1))

        self._iter += 1
        pred = self.O_Signal[edge_index[0]].max(dim = 1)[1]
        edges = torch.cat([edge_index[0][pred == 1].view(1, -1), edge_index[1][pred == 1].view(1, -1)], dim = 0)
        return self.forward(edges, N_pT, N_eta, N_phi, N_energy) 
        
   
    def message(self, x_i, x_j):
        edge_vector = x_i + x_j
        edge_mass = LV.MassFromPxPyPzE(edge_vector)
        score = self._mlp_m(edge_mass)
        return self._mlp(torch.cat([x_i, edge_vector, edge_mass, score], dim = 1))

    def deltaR(self, edge_index, eta, phi):
        d_eta = torch.pow(eta[edge_index[0]] - eta[edge_index[1]], 2)
        d_phi = torch.pow(phi[edge_index[0]] - phi[edge_index[1]], 2)
        return self._mlp_dr(torch.pow(d_eta + d_phi, 0.5))
        

