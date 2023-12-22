import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential as Seq, Linear
from torch.nn import Dropout

import pyc.Physics.Cartesian as physics
import pyc.Transform as transform
import pyc.Graph.Base as base_G


torch.set_printoptions(precision=3, sci_mode = True, linewidth = 1000)

def init_norm(m):
    if not type(m) == nn.Linear: return
    nn.init.uniform(m.weight)


class RecursiveParticles(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None)
        end = 32
        self.output = None
        self.rnn_edge = Seq(Linear(1, end), Linear(end, 2))

    def message(self, pmc, trk_j, pth_i):
        aggr_index = torch.cat([pth_i, trk_j], -1)

        pmc_ij = base_G.unique_aggregation(aggr_index, pmc)
        mass_ij = physics.M(pmc_ij).to(dtype = torch.float)
        deg = (aggr_index > -1).sum(-1).view(-1, 1)

        edge = self.rnn_edge(mass_ij)/deg
        return edge, edge.max(-1)[1]

    def aggregate(self, message, edge_index, pmc):
        edge_, edge_m = message
        edge_sel, edge_norm = edge_index[:, edge_m == 1], edge_.softmax(-1)[edge_m == 1][:, 1]

        val_idx, idx = to_dense_adj(edge_sel, edge_attr = edge_norm, max_num_nodes = pmc.size(0))[0].max(-1)
        no_null = val_idx > 0

        if self._mlp is None: self._mlp = edge_
        if not no_null.sum(-1): return self._mlp

        add_path  = torch.ones_like(idx)*-1
        add_gamma = torch.ones_like(val_idx)*-1

        add_path[no_null]  = idx[no_null]
        add_gamma[no_null] = val_idx[no_null]

        add_path  = add_path.view(-1, 1)
        add_gamma = add_gamma.view(-1, 1)

        msk = self._path[:, -1].view(-1, 1) == add_path
        add_path[msk] = -1

        self._path  = torch.cat([self._path , add_path ], -1)
        self._gamma = torch.cat([self._gamma, add_gamma], -1)

        deg = (self._path > -1).sum(-1)
        mass_prime = physics.M(base_G.unique_aggregation(self._path, pmc))

        msk = self._path.view(-1) > -1
        dst = self._path.view(-1)[msk]
        src = (torch.ones_like(self._path)*self._track).view(-1)[msk]

        idx_ = self._idx[src, dst]
        self._mlp[idx_] = self.rnn_edge(mass_prime[src].to(dtype = torch.float))/deg[src].view(-1, 1)

        self._idx[src, dst] = -1
        msk = self._idx[edge_index[0], edge_index[1]] > -1
        edge_prime = edge_index[:, msk]
        return self.propagate(edge_prime, pmc = pmc, trk = self._track, pth = self._path)


    def forward(self, edge_index, batch, pmc):

        track = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)
        self._track = track.clone()
        self._path = track.clone()
        self._gamma = torch.ones_like(track)*0.5

        self._idx = torch.cumsum(torch.ones_like(edge_index[0]), dim = -1)-1
        self._idx = to_dense_adj(edge_index, edge_attr = self._idx)[0]
        self._mlp = None
        self.output = self.propagate(edge_index, pmc = pmc, trk = self._track, pth = self._path)


class ExperimentalGNN(MessagePassing):

    def __init__(self):
        super().__init__()
        end = 128
        self.rnn_edge = RecursiveParticles()
        self.O_top_edge = None
        self.L_top_edge = "CEL"
        self._drop = Dropout(p = 0.5)

    def forward(self, edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep, N_pT, N_eta, N_phi, N_energy):
        pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        pmc = transform.PxPyPzE(pmu)
        self.rnn_edge(edge_index, batch, pmc)
        self.O_top_edge = self.rnn_edge.output
        self.O_top_edge = self._drop(self.O_top_edge)

