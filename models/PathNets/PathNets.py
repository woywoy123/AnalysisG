import torch
from torch import nn
from torch.nn import ReLU, Sigmoid, Sequential

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from PathNetOptimizerCUDA import AggregateIncomingEdges, Mass, ToDeltaR, ToPxPyPzE, ToPtEtaPhiE
from torch.distributions.bernoulli import Bernoulli

torch.set_printoptions(4, profile = "full", linewidth = 100000)

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2), Linear(x2, x2), ReLU(), Sigmoid()]
    return Seq(*out)

class PathNetsBase(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True 
        
        end = 256
        #self._eta = MakeMLP([1, end, end])
        #self._pt = MakeMLP([1, end, end])
        #self._energy = MakeMLP([1, end, end])
        #self._phi = MakeMLP([1, end, end])

        self._edgeM = Sequential(
                Linear(4, end), 
                Linear(end, end)
            )

        self._binary = Sequential(
                Linear(3, end), 
                Linear(end, 4*end), 
                Linear(4*end, end), 
                Linear(end, 2), 
            )

        self._it = 0
 
    def forward(self, i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi, E_T_edge):
 
        if self._it == 0:
            self._it = 1
            self._edgeindex = edge_index
            self.O_edge = self.forward(i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi, E_T_edge)
            self._it = 0
            return None
        self._it += 1

        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        ide = None #torch.cat([self._pt(N_phi), self._eta(N_eta), self._phi(N_phi), self._energy(N_energy)], dim = 1)

        O_edge = self.propagate(edge_index = edge_index, Pmu = Pmu, truth = E_T_edge)
        return O_edge

    def message(self, edge_index, Pmu_i, Pmu_j, truth):
        m_i = Mass(ToPxPyPzE(Pmu_i))
        m_j = Mass(ToPxPyPzE(Pmu_j))
        e_m = Mass(ToPxPyPzE(Pmu_j) + ToPxPyPzE(Pmu_i))
        return Pmu_j, m_i, m_j, e_m, truth 

    def aggregate(self, message, index, Pmu):
        Pmu_j, e_m, m_i, m_j, accept = message
        
        #mlp = self._binary(torch.cat([e_m, m_i, m_j], dim = 1))
        #accept = mlp.softmax(1).max(1)[1]
        
        m_node = AggregateIncomingEdges(Pmu_j, index.view(-1, 1), accept.view(-1, 1), True)
        bi = self._binary(torch.cat([accept, m_i-m_j, m_i], dim = 1))
        bi = nn.Conv1d(1, 2, kernel_size = (2, 1))(bi)
        return bi
