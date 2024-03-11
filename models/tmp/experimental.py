from torch_geometric.nn import MessagePassing, LayerNorm, aggr
from torch_geometric.utils import to_dense_adj

from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
from torch.nn import Dropout
import torch

import pyc
import pyc.Graph.Base as graph_base
import pyc.Graph.Cartesian as graph
import pyc.Transform as transform
import pyc.Physics.Cartesian as physics

torch.set_printoptions(precision=3, sci_mode = True, linewidth = 1000, threshold = 10000)

def init_norm(m):
    if not type(m) == torch.nn.Linear: return
    torch.nn.init.uniform(m.weight)


class RecursiveParticles(MessagePassing):

    def __init__(self, disable_nu = True, gev = False):
        super().__init__(aggr = None)
        self.output = None
        self._end = 32
        self._out = 2
        self._in = 7
        self._repeat = 7

        self.rnn_edge = Seq(
                Linear(self._in + self._repeat, self._end),
                LayerNorm(self._end),
                Linear(self._end, self._end),
                Tanh(),
                Linear(self._end, self._end),
                ReLU(),
                LayerNorm(self._end),
                Linear(self._end, self._repeat)
        )
        self.ret = Seq(Linear(self._repeat, self._out))

        self.rnn_edge.apply(init_norm)
        self.ret.apply(init_norm)

    def message(self, edge_index, pmc, trk_i, pmc_i, pmc_j):
        src, dst = edge_index
        idx = self._idx_mlp[src, dst]
        target = torch.cat([src.view(-1, 1), trk_i], -1)

        pmc_ij, pth = graph_base.unique_aggregation(target, pmc)
        iters = (pth > -1).sum(-1, keepdims = True)
        feats  = [physics.M(pmc_ij), iters]
        feats += [pmc_i - pmc_j, physics.DeltaR(pmc_i, pmc_j)]

        feats += [self._hidden[idx]]
        feats = torch.cat(feats, -1).to(dtype = torch.float)
        mlp = self.rnn_edge(feats)
        return mlp, self.ret(mlp).softmax(-1)

    def aggregate(self, message, edge_index, trk, pmc):
        mlp, mlp_ = message
        src, dst = edge_index
        sel = mlp_.max(-1)[1]

        idx = self._idx_mlp[dst, src]
        self._tmp = self._hidden.clone()
        self._tmp[idx] =  self._hidden[idx] +  (0.5 - sel.view(-1, 1))*mlp
        self._hidden = self._tmp
        if not self._iter: self._hidden = mlp
        self._iter += 1

        try: gr = graph.edge(edge_index, sel, pmc, True)[1]
        except KeyError: return self._hidden

        edge_index_ = edge_index[:, sel != 1]
        if not edge_index_.size(1): return self._hidden
        trk = gr["clusters"][gr["reverse_clusters"]]
        return self.propagate(edge_index_, pmc = pmc, trk = trk)

    def forward(self, edge_index, batch, pmc):
        src, _ = edge_index
        nulls  = torch.zeros_like(batch).view(-1, 1)
        feats  = torch.cat([physics.M(pmc), pmc], -1)
        null   = torch.cat([nulls for _ in range(self._in - feats.size(1))], -1)
        feats  = torch.cat([feats, null], -1)

        track = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)
        self._hidden = torch.cat([track]*(self._repeat), -1)
        self._hidden = torch.cat([feats, self._hidden], -1)
        self._hidden = self._hidden[src].clone()
        self._hidden = self._hidden.to(dtype = torch.float)
        self._hidden = self.rnn_edge(self._hidden)

        self._idx_mlp = torch.cumsum(torch.ones_like(src), dim = -1)-1
        self._idx_mlp = to_dense_adj(edge_index, edge_attr = self._idx_mlp)[0]

        self._iter = 0
        return self.ret(self.propagate(edge_index, pmc = pmc, trk = track))

class ExperimentalGNN(MessagePassing):

    def __init__(self):
        super().__init__(
                aggr = [
                    aggr.SoftmaxAggregation(learn = True),
                    aggr.MaxAggregation(),
                    aggr.VarAggregation()
                ]
        )

        try: dev = self.__param__["device"]
        except AttributeError: dev = "cuda"
        try: self._gev = self.__param__["gev"]
        except AttributeError: self._gev = False

        self._n = 64
        self.coder = Seq(
                Linear(2 + 32, self._n),
                LayerNorm(self._n),
                Linear(self._n, self._n), Tanh(),
                Linear(self._n, self._n), ReLU(),
                Linear(self._n, self._n)
        )
        self.ntops = Seq(Linear(self._n*3, 5))

        self.rnn_edge = RecursiveParticles(True, self._gev)
        self.rnn_edge.to(device = dev)

        self.O_top_edge = None
        self.L_top_edge = "CEL"

        self.O_ntops = None
        self.L_ntops = "CEL"

        self._drop = Dropout(p = 0.01)
        self._pool = aggr.SoftmaxAggregation(learn = True)


    def message(self, tope, data_i, data_j):
        tmp = torch.cat([tope, data_i, data_i - data_j], -1)
        return self.coder(tmp.to(dtype = torch.float))

    def forward(self,
            edge_index, batch,
            G_met, G_phi, G_n_jets, G_n_lep,
            N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b
        ):

        pid    = torch.cat([N_is_lep, N_is_b], -1)
        met_x  = transform.Px(G_met, G_phi)
        met_y  = transform.Py(G_met, G_phi)
        met_xy = torch.cat([met_x, met_y], -1)
        pmu    = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        pmc    = transform.PxPyPzE(pmu)
        grph   = torch.cat([G_n_jets, G_n_lep], -1)

        if not self._gev: pass
        else: pmc, met_xy = pmc/1000, met_xy/1000

        out = self.rnn_edge(edge_index, batch, pmc)
        try: gr = graph.edge(edge_index, out.max(-1)[1], pmc, True)[1]["node_sum"]
        except KeyError: gr = torch.zeros_like(pmc)

        data = torch.cat([pmc, gr - pmc, gr, pid, pmc[:, :2] - met_xy[batch]], -1)
        aggre = self.propagate(edge_index, tope = out, data = data)
        aggre = self._pool(aggre, batch)
        aggre = self.ntops(aggre)
        self.O_top_edge = self._drop(out)
        self.O_ntops = self._drop(aggre)
        self.iter = self.rnn_edge._iter

