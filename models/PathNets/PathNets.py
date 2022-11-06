import torch
from torch.nn import Sequential as Seq

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from LorentzVector import *

from PathNetOptimizer import PathCombinatorial
from PathNetOptimizerCUDA import PathVector, PathMass, IncomingEdgeVector, IncomingEdgeMass

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2)]
    return Seq(*out)

class PathNetsTruthJet(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source") 

        #self.O_from_top = None
        #self.L_from_top = "CEL"
        #self.C_from_top = True

        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True

        self.max = 5
        self._n = 0

        self._edge = MakeMLP([1, 4096, 4096, 2])
        self._AdjMatrix = None

    def forward(self, i, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi):
        device = edge_index.device
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmu = TensorToPxPyPzE(Pmu)
        Mass = MassFromPxPyPzE(Pmu)/1000

        if int(num_nodes) != self._n or self._AdjMatrix == None:
            self._n = int(num_nodes)
            self._AdjMatrix = PathCombinatorial(self._n, self.max, str(device).split(":")[0])
        self.propagate(edge_index, Pmu = Pmu)

    def message(self, edge_index, Pmu_i, Pmu_j):
        return Pmu_j

    def aggregate(self, message, index, Pmu):
        print(index.view(-1, self._n))
        
        V, adj = IncomingEdgeVector(self._AdjMatrix, message, index.view(-1, 1))
        M = MassFromPxPyPzE(V)/1000
        print(self._AdjMatrix[adj].view(-1, self._n), M)
        




