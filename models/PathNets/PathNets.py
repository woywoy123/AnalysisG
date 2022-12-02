import torch
from torch.nn import Sequential as Seq
from torch.nn import ReLU, Sigmoid

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from PathNetOptimizerCUDA import AggregateIncomingEdges, Mass, ToDeltaR, ToPxPyPzE, ToPtEtaPhiE
from torch.distributions.bernoulli import Bernoulli

torch.set_printoptions(4, profile = "full", linewidth = 100000)

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2), Linear(x2, x2)]
    return Seq(*out)

class PathNetsBase(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True 
        
        end = 1024
        self._eta = MakeMLP([1, 64, end])
        self._pt = MakeMLP([1, 64, end])
        self._energy = MakeMLP([1, 64, end])
        self._phi = MakeMLP([1, 64, end])
        self._mass = MakeMLP([1, 64, end])
        
        self._edgeM = MakeMLP([end*3, 64, 2])
        self._edgeK = MakeMLP([end*4*3, 64, 2])

        self._it = 0
 
    def forward(self, i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi):
 
        if self._it == 0:
            self._it = 1
            self._edgeindex = edge_index
            self.O_edge = self.forward(i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi)
            self._it = 0
            return self.O_edge
        self._it += 1

        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        ide = torch.cat([self._pt(N_phi), self._eta(N_eta), self._phi(N_phi), self._energy(N_energy)], dim = 1)
        O_edge = self.propagate(edge_index = edge_index, Pmu = Pmu, ide = ide)

        index = O_edge.max(1)[1] #
        #e_i = edge_index[0][index == 1].view(1, -1)
        #e_j = edge_index[1][index == 1].view(1, -1)
        #n_edge_index = torch.cat([e_i, e_j], dim = 0)
        
        if torch.equal(index, self._edgeindex): # or self._it > 2:
            return O_edge
        self._edgeindex = index
       
        #PMC = ToPxPyPzE(Pmu)
        #PMC[e_i] += PMC[e_j]
        #
        #Pmu = ToPtEtaPhiE(PMC)
        #N_pT = Pmu[:, 0].view(-1, 1)
        #N_eta = Pmu[:, 1].view(-1, 1)
        #N_phi = Pmu[:, 2].view(-1, 1)
        #N_energy = Pmu[:, 3].view(-1, 1)
    
        return O_edge + self.forward(i, batch, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi)

    def message(self, edge_index, Pmu_i, Pmu_j, ide_i, ide_j):
        m_i = self._mass(Mass(ToPxPyPzE(Pmu_i)))
        m_j = self._mass(Mass(ToPxPyPzE(Pmu_j)))
        e_m = self._mass(Mass(ToPxPyPzE(Pmu_j) + ToPxPyPzE(Pmu_i)))
         
        mlp = self._edgeM(torch.cat([m_i, e_m, m_j], dim = 1))
        mk = self._edgeK(torch.cat([ide_i, ide_j, torch.abs(ide_i - ide_j)], dim = 1))
        
        return Pmu_j, mlp + mk, m_i, m_j

    def aggregate(self, message, index, Pmu):
        Pmu_j, Pedge, m_i, m_j = message
        
        accept = Bernoulli(Pedge.softmax(1)[:, 1]).sample().to(dtype = torch.int32) #Pedge.softmax(1).max(1)[1].view(-1, 1)
        m_pred = self._mass(AggregateIncomingEdges(Pmu_j, index.view(-1, 1), accept.view(-1, 1), True)).softmax(dim = 1)
        mall = self._edgeM(torch.cat([m_i, m_pred[index], m_j], dim = 1))
        
        return mall + Pedge
