import torch
from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.nn.models import MLP
from torch_geometric.utils import remove_self_loops

try: import PyC.Transform.CUDA as Tr
except: import PyC.Transform.Tensors as Tr

try: import PyC.Physics.CUDA.Polar as PP
except: import PyC.Physics.Tensors.Polar as PP

try: import PyC.Physics.CUDA.Cartesian as PC
except: import PyC.Physics.Tensors.Cartesian as PC

try: import PyC.Operators.CUDA as OP
except: import PyC.Operators.Tensors as OP

try: import PyC.NuSol.CUDA as NuSol
except: import PyC.NuSol.Tensor as NuSol

torch.set_printoptions(4, profile = "full", linewidth = 100000)

from time import sleep
from torch_geometric.utils import to_dense_adj


class ParticleRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        end = 32

        self.edge = None

        ef = 3
        self._norm_i = LayerNorm(ef, mode = "node")
        self._inpt = MLP([ef, end])
        self._edge = MLP([end, 2])

        self._norm_r = LayerNorm(end, mode = "node")
        self._rnn = MLP([end*2, end*2, end*2, end]) 
    
    def edge_updater(self, edge_index, batch, Pmc, Pmu, PiD): 
        i, j = edge_index[0], edge_index[1]
        mlp, _ = self.message(batch[i], batch[j], Pmc[i], Pmc[j], Pmu[i], Pmu[j], PiD[i], PiD[j])
        if self.edge is None: self.edge = self._norm_r(mlp); return 

    def message(self, batch_i, batch_j, Pmc_i, Pmc_j, Pmu_i, Pmu_j, PiD_i, PiD_j):
        m_i, m_j, m_ij = PC.Mass(Pmc_i), PC.Mass(Pmc_j), PC.Mass(Pmc_i + Pmc_j)
        f_ij = torch.cat([m_i + m_j, m_ij, torch.abs(m_i - m_j)], -1)
        f_ij = f_ij + self._norm_i(f_ij)
        return self._inpt(f_ij), Pmc_j 

    def aggregate(self, message, index, Pmc):
        mlp_ij, Pmc_j = message
      
        msk = self._idx == 1 
        torch.cat([mlp_ij, self.edge[msk]], -1)
        self.edge[msk] = self._norm_r(self._rnn(torch.cat([mlp_ij, self.edge[msk]], -1)))
        
        sel = self._edge(self.edge[msk])
        sel = sel.max(-1)[1]
        self._idx[msk] *= (sel == 0).to(dtype = torch.int)

        Pmc = Pmc.clone()
        Pmc.index_add_(0, index[sel == 1], Pmc_j[sel == 1])
        Pmu = Tr.PtEtaPhi(Pmc[:, 0].view(-1, 1), Pmc[:, 1].view(-1, 1), Pmc[:, 2].view(-1, 1))
        Pmu = torch.cat([Pmu, Pmc[:, 3].view(-1, 1)], -1)
        return sel, Pmc, Pmu

    def forward(self, i, edge_index, batch, Pmc, Pmu, PiD):

        if self.edge is None:
            self._idx = torch.ones_like(edge_index[0])
            self._idx *= edge_index[0] != edge_index[1]
            self.edge_updater(edge_index, batch = batch, Pmc = Pmc, Pmu = Pmu, PiD = PiD)
            edge_index, _ = remove_self_loops(edge_index)

        _idx = self._idx.clone()        
        sel, Pmc, Pmu = self.propagate(edge_index, batch = batch, Pmc = Pmc, Pmu = Pmu, PiD = PiD)
        edge_index = edge_index[:, sel == 0]       
        if edge_index.size()[1] == 0: pass
        elif (_idx != self._idx).sum(-1) != 0: return self.forward(i, edge_index, batch, Pmc, Pmu, PiD)
        mlp = self._edge(self.edge) 
        self.edge = None 
        return mlp

class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):
        super().__init__( aggr = None, flow = "target_to_source")
        
        self.O_top_edge = None 
        self.L_top_edge = "CEL"
        self._edgeRNN = ParticleRecursion()

    def message(self, edge_index, Pmc_i, Pmc_j, PiD_i, PiD_j):
        # Find edges where the source/dest are a b and lep
        _lep_b = ((PiD_i + PiD_j) == 1).sum(-1) == 2
        _lep_b = _lep_b == 1
        c_t, s_t = OP.CosTheta(Pmc_i, Pmc_j), OP.SinTheta(Pmc_i, Pmc_j)
        e_feat = torch.cat([c_t, s_t, _lep_b.view(-1, 1)], -1)
        return e_feat, _lep_b, Pmc_j, edge_index[1]

    def aggregate(self, message, index, batch, Pmc, PiD, met, phi):
        e_feat, msk, Pmc_j, index_j = message
        e_ij = index[msk]
        
        this_lep = Pmc[e_ij]  * PiD[e_ij][:, 0].view(-1, 1)
        this_b   = Pmc_j[msk] * (PiD[e_ij][:, 1] == 0).view(-1, 1)

        matrix = torch.cat([this_lep, this_b, index[msk].view(-1, 1), index_j[msk].view(-1, 1)], -1)
        this_matrix = matrix[(matrix != 0).sum(-1) > 3]

        e_ij = this_matrix[:, -2:]
        a, a_ = e_ij[:, 0].unique(sorted = False, return_inverse = True)
        b, b_ = e_ij[:, 1].unique(sorted = False, return_inverse = True)
        e_ij = e_ij[a_ == b_]

        li_bj_node = this_matrix[:, :-2].view(-1, 8)
        li_bj_node = li_bj_node[a_ == b_]





        exit()




 
    def forward(self, i, edge_index, batch, G_met, G_phi, G_n_jets, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b):
        Pmc = torch.cat([Tr.PxPyPz(N_pT, N_eta, N_phi), N_energy], -1)
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        PiD = torch.cat([N_is_lep, N_is_b], -1)
        batch = batch.view(-1, 1)

        self.propagate(edge_index, batch = batch, Pmc = Pmc, PiD = PiD, met = G_met, phi = G_phi)
        
        self.O_top_edge = self._edgeRNN(i, edge_index, batch, Pmc, Pmu, PiD)
