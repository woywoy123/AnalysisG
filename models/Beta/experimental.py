from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
from torch.nn import Dropout
import torch

import pyc.Transform as transform
import pyc.NuSol.Cartesian as nusol
import pyc.Graph.Base as graph_base
import pyc.Physics.Cartesian as physics
torch.set_printoptions(precision=3, sci_mode = True, linewidth = 1000, threshold = 10000)

def init_norm(m):
    if not type(m) == torch.nn.Linear: return
    torch.nn.init.uniform(m.weight)


class RecursiveParticles(MessagePassing):

    def __init__(self, disable_nu = True, gev = False):
        super().__init__(aggr = None)
        self.output = None
        self._end = 1024
        self.rnn_edge = Seq(Linear(2, self._end), Linear(self._end, 2))
        self.rnn_edge.apply(init_norm)

    def message(self, pth_j, trk_i, trk_j):
        pth = torch.cat([pth_j, trk_i], -1)
        pmc_ij, pth = graph_base.unique_aggregation(pth, self._pmc)

        # features
        mass_ij = physics.M(pmc_ij)
        iters_ij = (pth > -1).sum(-1, keepdim = True)

        feats = [mass_ij, iters_ij]
        feats = torch.cat(feats, -1).to(dtype = torch.float)
        return self.rnn_edge(feats)

    def aggregate(self, message, edge_index):
        if self._pmc.size(0) < self._iter: return self._mlp
        self._mlp = message
        msk = self._msk*(self._mlp.max(-1)[1] > 0)
        sft = self._mlp[msk][:, 1].detach()
        pn = to_dense_adj(edge_index[:, msk], max_num_nodes = self._max_n)[0]
        p  = to_dense_adj(edge_index[:, msk], edge_attr = sft, max_num_nodes = self._max_n)[0]
        p  = p.softmax(-1)*pn
        msk_ = p.sum(-1) > 0

        idx = p[msk_].max(-1)[1].view(-1, 1)
        x = self._idx[self._track[msk_].view(-1), idx.view(-1)]
        self._msk[x] = False

        if not msk_.sum(-1): return self._mlp
        if not self._msk.sum(-1): return self._mlp
        nulls = torch.ones_like(self._track)*-1
        nulls[msk_] = idx
        pth = torch.cat([self._path, nulls], -1)
        self._iter += 1
        return self.propagate(edge_index = edge_index, pth = pth, trk = self._track)

    def forward(self, edge_index, batch, pmc, pid, met_xy):
        src, dst    = edge_index
        self._max_n = batch.size(0)
        self._iter  = 0
        self._pmc   = pmc

        self._msk   = edge_index[0] > -1
        self._mlp   = torch.zeros_like(edge_index.view(-1, 2)).to(dtype = torch.float)
        self._idx   = (torch.ones_like(edge_index[0]).cumsum(-1) -1)
        self._idx   = to_dense_adj(edge_index, edge_attr = self._idx)[0]
        self._path  = (torch.ones_like(batch).cumsum(-1) - 1).view(-1, 1)
        self._track = self._path.clone()
        self.output = self.propagate(edge_index = edge_index, pth = self._path, trk = self._track)

class ExperimentalGNN(MessagePassing):

    def __init__(self):
        super().__init__()
        self._gev = False
        self.rnn_edge = RecursiveParticles(True, self._gev)
        self.O_top_edge = None
        self.L_top_edge = "CEL"
        self._drop = Dropout(p = 0.01)

    def forward(self,
            edge_index, batch,
            G_met, G_phi, G_n_jets, G_n_lep,
            N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b
        ):

        pid = torch.cat([N_is_lep, N_is_b], -1)
        met_x = transform.Px(G_met, G_phi)
        met_y = transform.Py(G_met, G_phi)
        pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)

        pmc = transform.PxPyPzE(pmu)
        met_xy = torch.cat([met_x, met_y], -1)
        if self._gev: f = 1/1000
        else: f = 1

        pmc = pmc*f
        met_xy = met_xy*f
        self.rnn_edge(edge_index, batch, pmc, pid, met_xy)
        self.O_top_edge = self.rnn_edge.output
        self.iter = self.rnn_edge._iter
