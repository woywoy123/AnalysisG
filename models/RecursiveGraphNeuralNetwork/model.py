import torch
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import scatter
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
import torch.nn.functional as F

try: import PyC.Transform.CUDA as Tr
except: import PyC.Transform.Tensors as Tr

try: import PyC.Physics.CUDA.Polar as PP
except: import PyC.Physics.Tensors.Polar as PP

try: import PyC.Physics.CUDA.Cartesian as PC
except: import PyC.Physics.Tensors.Cartesian as PC

torch.set_printoptions(4, profile = "full", linewidth = 100000)

from time import sleep
from torch_geometric.utils import to_dense_adj


class ParticleRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        end = 64
       
        self.edge = None 
        self._isEdge = Seq(Linear(2, end), Linear(end, end), Linear(end, 2))
        self._rnn = Seq(Linear(6, end), Linear(end, end), Linear(end, 2))
 
    def message(self, edge_index, Pmc_i, Pmc_j):
        excl_self = edge_index[0] != edge_index[1]
        m_i, m_j = PC.Mass(Pmc_i), PC.Mass(Pmc_j)
        m_ij = PC.Mass(Pmc_i + Pmc_j*excl_self.view(-1, 1))
        mlp_ij = self._isEdge(torch.cat([m_ij, self._tmp], -1))
        return mlp_ij, excl_self, Pmc_j
 
    def aggregate(self, message, index, Pmc):
        mlp_ij, excl, pmc_j = message
        excl *= self._idx.to(dtype = torch.bool)
        
        # Make a MLP prediction on which edges are to be connected
        sel = mlp_ij.max(-1)[1]
        excl *= sel.to(dtype = torch.bool)
        
        # Sum incoming edges provided they are predicted to be connected and not self-loops
        self._pmc[index[excl]] += pmc_j[excl]

        print(PC.Mass(self._pmc))

        # Make a new prediction using the RNN
        mlp_n = self._isEdge(torch.cat([PC.Mass(self._pmc[index]), self._tmp], -1))

        if self.edge is None: self.edge = self._isEdge(torch.cat([PC.Mass(Pmc[index]), self._tmp], -1))
        self.edge[excl] = self._rnn(torch.cat([self.edge, mlp_ij, mlp_n], -1))[excl]

        # Update the unused edges and try them again 
        self._idx = excl
        return self._pmc, self.edge

    def forward(self, i, edge_index, N_pT, N_eta, N_phi, N_energy, E_T_top_edge):
        Pmc = torch.cat([Tr.PxPyPz(N_pT, N_eta, N_phi), N_energy], -1)

        if self.edge is None: 
            self._idx = torch.ones_like(edge_index[0])        
            self._pmc = Pmc.clone()
            self._tmp = E_T_top_edge

        _idx = self._idx.clone()
        Pmc_n, mlp_pred = self.propagate(edge_index, Pmc = Pmc)
        if self._idx.sum(-1) == 0: return self.edge
        if (_idx != self._idx).sum(-1) == 0: return self.edge
        
        #_px, _py, _pz = self._pmc[:, 0], self._pmc[:, 1], self._pmc[:, 2]
        #_pmu = Tr.PtEtaPhi(_px.view(-1, 1), _py.view(-1, 1), _pz.view(-1, 1))
        #_pT, _eta, _phi = _pmu[:, 0].view(-1, 1), _pmu[:, 1].view(-1, 1), _pmu[:, 2].view(-1, 1)
        #_en = self._pmc[:, 3].view(-1, 1) 
        return self.forward(i, edge_index, N_pT, N_eta, N_phi, N_energy, E_T_top_edge)


class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        self.O_top_edge = None 
        self.L_top_edge = "CEL"
        
        #self.O_res_edge = None 
        #self.L_res_edge = "CEL"
        
        self._isEdge = ParticleRecursion()


    def forward(self, i, edge_index, N_pT, N_eta, N_phi, N_energy, E_T_top_edge):
        self._isEdge.edge = None 
        self.O_top_edge = self._isEdge(i, edge_index, N_pT/1000, N_eta, N_phi, N_energy/1000, E_T_top_edge)   
        
        print(to_dense_adj(edge_index, edge_attr = self.O_top_edge.max(-1)[1]))

