import torch
from torch.nn import Sequential as Seq

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from LorentzVector import *
from PathNetOptimizerCUDA import *
from PathNetOptimizerCPU import CombinatorialCPU

torch.set_printoptions(4, profile = "full", linewidth = 100000)

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2)]
    return Seq(*out)

class PathNetsBase(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True 
        
        self._mass = MakeMLP([1, 128, 2])
    
    def forward(self, i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi, E_T_edge):
        Pmu = TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)) 
        
        self._adjM = CombinatorialCPU(num_nodes.item(), 4, "cuda")
        out = self.propagate(edge_index = edge_index, Pmu = Pmu)
        self.O_edge = out 
        return out

    def message(self, edge_index, Pmu_i, Pmu_j):
        return Pmu_j, edge_index

    def aggregate(self, message, index, Pmu):
        Pmu_j, edge_index = message
        mass, adj = IncomingEdgeMassCUDA(self._adjM, Pmu_j, index.view(-1, 1))
        _edge = self._adjM[adj].view(-1, Pmu.size()[0])
        _mass = self._mass(mass/1000)
        
        print(_mass)
        print(_mass[:, 0].view(-1, 1))
        _zero = _mass[:, 0].view(-1, 1)*_edge
        _one = _mass[:, 1].view(-1, 1)*_edge
        
        mx = adj.max(0)[0]+1
        

        exit()

        _zero = _zero.view(-1, mx, Pmu.size()[0]).sum(1)
        _one = _one.view(-1, mx, Pmu.size()[0]).sum(1)


        print(_zero)
        print(_one) 
     
        return torch.cat([_zero[index, edge_index[1]].view(-1, 1), _one[index, edge_index[1]].view(-1, 1)], dim = 1)
