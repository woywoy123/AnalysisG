import torch
from torch.nn import Sequential as Seq
from torch.nn import ReLU, Sigmoid

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from LorentzVector import *
from PathNetOptimizerCUDA import AggregateIncomingEdges, Mass, ToDeltaR, ToPxPyPzE
from torch.distributions.bernoulli import Bernoulli

torch.set_printoptions(4, profile = "full", linewidth = 100000)

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2), ReLU(), Linear(x2, x2), Sigmoid()]
    return Seq(*out)

class PathNetsBase(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True 
        
        end = 1024
        self._mass = MakeMLP([1, end, 64, 64, end])
        self._identity = MakeMLP([2, end])
        self._Bern = MakeMLP([end*2, 2])
        self._nodeMass = MakeMLP([end*3+2, 128, 128, 2])
    
    def forward(self, i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi, E_T_edge):
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        out = self.propagate(edge_index = edge_index, Pmu = Pmu)

        self.O_edge = out 
        return out

    def message(self, edge_index, Pmu_i, Pmu_j):
        idx = self._identity(torch.cat([Mass(ToPxPyPzE(Pmu_i)), Mass(ToPxPyPzE(Pmu_j))], dim = 1))
        edge = self._mass(Mass(ToPxPyPzE(Pmu_i) + ToPxPyPzE(Pmu_j)))
        return Pmu_j, edge, idx

    def aggregate(self, message, index, Pmu):
        Pmu_j, Pedge, idx = message
        
        ber = self._Bern(torch.cat([Pedge, idx], dim = 1)).softmax(1)
        accept = Bernoulli(ber[:, 1]).sample().to(dtype = torch.int32).view(-1, 1)
        pred = self._mass(AggregateIncomingEdges(Pmu_j, index.view(-1, 1), accept, True)).softmax(dim = 1)
        
        return self._nodeMass(torch.cat([Pedge, idx, pred[index], ber], dim = 1))
