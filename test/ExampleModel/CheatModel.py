import torch 
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
import torch.nn.functional as F
from LorentzVector import *
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops, dense_to_sparse

torch.set_printoptions(4, profile = "full", linewidth = 100000)

class CheatModel(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True

        end = 16
        self._isEdge = Seq(Linear(end*3 +1, end), ReLU(), Linear(end, 2))
        self._isMass = Seq(Linear(1, end), Linear(end, end))
        self._it = 0

    def forward(self, i, edge_index, N_pT, N_eta, N_phi, N_energy, N_mass, E_T_edge):
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmc = TensorToPxPyPzE(Pmu)
        
        self.O_edge = self.propagate(edge_index, Pmc = Pmc, Pmu = Pmu, Mass = N_mass, E_T_edge = E_T_edge)
        return self.O_edge

    def message(self, edge_index, Pmc_i, Pmc_j, Pmu_i, Pmu_j, Mass_i, Mass_j, E_T_edge):
        e_dr = TensorDeltaR(Pmu_i, Pmu_j)
        e_mass = MassFromPxPyPzE(Pmc_i + Pmc_j)

        e_mass_mlp = self._isMass(e_mass/1000)
        ni_mass = self._isMass(Mass_i/1000)
        nj_mass = self._isMass(Mass_j/1000)

        mlp = self._isEdge(torch.cat([e_mass_mlp, torch.abs(ni_mass-nj_mass), torch.abs(e_mass_mlp - ni_mass - nj_mass), E_T_edge], dim = 1))
        return edge_index[1], mlp, Pmc_j

    def aggregate(self, message, index, Pmc, Pmu, Mass):
        edge_index, mlp_mass, Pmc_j = message
        return mlp_mass
