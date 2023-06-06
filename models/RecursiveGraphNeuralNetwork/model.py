import torch
from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.utils import scatter
from torch.nn import Sequential as Seq
from torch.nn import Linear, ReLU, Sigmoid, Tanh

import torch.nn.functional as F

try: import PyC.Transform.CUDA as Tr
except: import PyC.Transform.Tensors as Tr

try: import PyC.Physics.CUDA.Polar as PP
except: import PyC.Physics.Tensors.Polar as PP

try: import PyC.Physics.CUDA.Cartesian as PC
except: import PyC.Physics.Tensors.Cartesian as PC

try: import PyC.Operators.CUDA as OP
except: import PyC.Operators.Tensors as OP


torch.set_printoptions(4, profile = "full", linewidth = 100000)

from time import sleep
from torch_geometric.utils import to_dense_adj


class ParticleRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        end = 16
      
        self._it = 0 
        self.edge = None 
        self._norm = LayerNorm(16)
        self._isEdge = Seq(Linear(16,  end), Linear(end, end), Linear(end, 16))

        self._norm_m = LayerNorm(5)
        self._isMass = Seq(Linear(5, end), Linear(end, end), Linear(end, 5))

        self._norm_r = LayerNorm(23)
        self._rnn = Seq(Linear(23, end), Linear(end, end), Linear(end, 2))

    def M(self, pmc): return PC.Mass(pmc)
 
    def message(self, edge_index, Pmc_i, Pmc_j, Pmu_i, Pmu_j, type__i, type__j):
        eta_i, eta_j = Pmu_i[:, 1].view(-1, 1), Pmu_j[:, 1].view(-1, 1)
        phi_i, phi_j = Pmu_i[:, 2].view(-1, 1), Pmu_j[:, 2].view(-1, 1)
        delta_r = PP.DeltaR(eta_i, eta_j, phi_i, phi_j)
       
        idx = (edge_index[0] != edge_index[1]).view(-1, 1)
        m_i, m_j = PC.Mass(Pmc_i), PC.Mass(Pmc_j)
        m_ij = PC.Mass(Pmc_i + Pmc_j*idx)

        feat_i, feat_j = torch.cat([Pmc_i, type__i, m_i], -1), torch.cat([Pmc_j, type__j, m_j], -1)
        tmp = self._norm(torch.cat([delta_r, m_ij, feat_i - feat_j, feat_i], -1))
        
        mlp = self._isEdge(tmp) + tmp
        return mlp, Pmc_j
 
    def aggregate(self, message, index, Pmc, type_, batch):
        mlp_ij, pmc_j = message
       
        # Make a new prediction of the node mass
        pmc ,  _pmc,  __pmc =  Pmc[index], self._pmc[index], self._pmc[index] + pmc_j
        mass, _mass, __mass = self.M(pmc),     self.M(_pmc), self.M(__pmc)
        batch = batch[index]

        masses = torch.cat([mass - _mass, mass, _mass - __mass, _mass, __mass], -1)
        masses = self._norm_m(masses, batch = batch)
        mlp_m = self._isMass(masses) + masses

        inpt = [mass, _mass] if self.edge is None else [self.edge]
        inpt += [mlp_m, mlp_ij]
        inpt = torch.cat(inpt, -1)
        self.edge = self._rnn(self._norm_r(inpt, batch = batch))
 
        if self.edge is None: self.edge = self._rnn()
        else: self.edge = self._rnn(torch.cat([self.edge, mlp_m, mlp_ij], -1)) 
 
        # Make a MLP prediction on which edges are to be connected
        sel = self.edge.max(-1)[1]
        msk = (sel*self._idx) == 1
        
        # Sum incoming edges provided they are predicted to be connected and not self-loops
        self._pmc[index[msk]] += pmc_j[msk]
        
        # Update the unused edges and try them again 
        self._idx *= (sel == 0).to(dtype = torch.int)

        return self._pmc, self.edge

    def forward(self, i, edge_index, batch, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b):
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        Pmc = torch.cat([Tr.PxPyPz(N_pT, N_eta, N_phi), N_energy], -1)
        type_ = Misc = torch.cat([N_is_lep, N_is_b], -1)
        
        if self.edge is None: 
            self._idx = torch.ones_like(edge_index[0])
            self._idx *= (edge_index[0] != edge_index[1]) # Prevent double counting 
            self._pmc = Pmc.clone()
            self._it = 1

        _idx = self._idx.clone()
        Pmc_n, mlp_pred = self.propagate(edge_index, Pmc = Pmc, Pmu = Pmu, type_ = type_, batch = batch)
        if (self._idx - _idx).sum(-1) == 0: return mlp_pred
        self._it += 1
        return self.forward(i, edge_index, batch, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b)


class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):
        super().__init__( aggr = "max" )
        
        self.O_top_edge = None 
        self.L_top_edge = "CEL"
        
        self.O_res_edge = None 
        self.L_res_edge = "CEL"

        self.O_signal = None 
        self.L_signal = "CEL"
    
        self.O_ntops = None
        self.L_ntops = "CEL"       
 
        self._istop = ParticleRecursion()
        self._isres = ParticleRecursion()

        self._mlp = Seq(Linear(12, 12), Linear(12, 4))
        self._norm = LayerNorm(8)
        self._mlp_n = Seq(Linear(8, 8), LayerNorm(8), Linear(8, 8), Linear(8, 5))
        self._mlp_s = Seq(Linear(8, 8), LayerNorm(8), Linear(8, 8), Linear(8, 2))


    def message(self, Pmc_i, Pmc_j):
        return self._mlp( torch.cat([Pmc_i, Pmc_j - Pmc_i], -1) )

    def forward(self, i, edge_index, batch, G_met, G_phi, G_n_jets, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b, E_T_res_edge):

        self._istop.edge = None 
        self._isres.edge = None
        self.O_top_edge = self._istop(i, edge_index, batch, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b)   
        self.O_res_edge = self._isres(i, edge_index, batch, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b)   
        dot = OP.Dot(self.O_res_edge, self.O_top_edge)
        self.O_res_edge = dot*self.O_res_edge + (1-dot)*self.O_top_edge

        Pmc = torch.cat([Tr.PxPyPz(N_pT, N_eta, N_phi), N_energy, N_is_lep, N_is_b], -1)
        conv = self.propagate(edge_index, Pmc = Pmc)

        px_, py_ = Tr.Px(G_met, G_phi), Tr.Py(G_met, G_phi)
        dx, dy = px_[batch] - Pmc[:, 0].view(-1, 1), py_[batch] - Pmc[:, 1].view(-1, 1)
        conv = torch.cat([conv, torch.sqrt(dx.pow(2) + dy.pow(2)), G_n_jets[batch], N_is_lep, N_is_b], -1)
        conv = self._norm(conv, batch = batch) + conv

        ntop = torch.zeros_like(G_met)          
        ntop = torch.cat([ntop for _ in torch.arange(8)], -1)
        ntop[batch] += conv
        self.O_ntops = self._mlp_n(ntop)

        sig = torch.zeros_like(G_met)          
        sig = torch.cat([sig for _ in torch.arange(8)], -1)
        sig[batch] += conv
        self.O_signal = self._mlp_s(sig) 
        
 
        
