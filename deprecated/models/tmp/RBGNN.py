import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh

import pyc
import pyc.Graph.Base as base
import pyc.Transform as transform
import pyc.Graph.Cartesian as graph
import pyc.Physics.Cartesian as physics

torch.set_printoptions(precision=3, sci_mode = True)

def init_norm(m):
    if not type(m) == nn.Linear: return
    nn.init.uniform(m.weight)


class RecursiveMarkovianGraphNet(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None)
        self._path = Seq(Linear(1, 32), Tanh(), Linear(32, 2))
        self._path.apply(init_norm)


    def message(self, pmc, pmc_i, pmc_j, trk_i, trk_j):
        trk_ = torch.cat([trk_j, trk_i], -1)
        pmc_ij, _ = base.unique_aggregation(trk_, pmc)
        mass_ij = physics.M(pmc_ij).to(dtype = torch.float)
        mlp_ij = nn.Softmax(-1)(self._path(mass_ij))
        o_ij = torch.cat([trk_j.view(1, -1), trk_i.view(1, -1)], dim = 0)
        return mass_ij, o_ij, mlp_ij

    def aggregate(self, message, pmc):
        mass_ij, o_ij, mlp_ij = message

        # Probability that a given edge is either 0 or 1 
        g00, g11 = mlp_ij[:, 0], mlp_ij[:, 1]
        g00_ = to_dense_adj(o_ij, edge_attr = g00)[0]
        g11_ = to_dense_adj(o_ij, edge_attr = g11)[0]

        # Normalized probability that the given edge is chosen and is 0/1.
        G00 = to_dense_adj(o_ij, edge_attr = g00)[0]
        G00 = G00/G00.sum(-1).view(-1, 1)

        G11 = to_dense_adj(o_ij, edge_attr = g11)[0]
        G11 = G11/G11.sum(-1).view(-1, 1)

        # Transition probability of selecting/not-selecting an edge, given it is actually true.
        # P(E | ET) && P(NE | NET)
        P00, P11 = G00*g00_, G11*g11_

        # Transition probability of selecting/not-selecting an edge, given it is actually false.
        # P(E | NET) && P(NE | ET)
        P01, P10 = G00*g11_, G11*g00_

        return pmc_i

    def forward(self, edge_index, batch, N_pT, N_eta, N_phi, N_energy):
        pmu = torch.cat([N_pT/1000, N_eta, N_phi, N_energy/1000], -1)
        pmc = transform.PxPyPzE(pmu)

        self._idx = edge_index[0, edge_index[0] == edge_index[1]].view(-1, 1)
        self._idx = torch.ones_like(self._idx)
        self._idx = self._idx.cumsum(0)-1

        self.propagate(edge_index, pmc = pmc, trk = self._idx)


        exit()


