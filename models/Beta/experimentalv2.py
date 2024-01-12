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

def addf(i, j): return [i, i - j]

class RecursiveParticles(MessagePassing):

    def __init__(self, disable_nu = True, gev = False):
        super().__init__(aggr = None)
        self.output = None
        self._end = 128
        self._out = 2
        self._in = 44
        self._repeat = 20

        self.rnn_edge = Seq(
                Linear(self._in + self._repeat, self._end),
                LayerNorm(self._end), ReLU(), Linear(self._end, self._end), Tanh(),
                LayerNorm(self._end), ReLU(), Linear(self._end, self._end), Tanh(),
                Linear(self._end, self._repeat)
        )

        self.ret = Seq(Linear(self._repeat, self._out))
        self.rnn_edge.apply(init_norm)
        self.ret.apply(init_norm)

    def message(self, edge_index, pmc, trk_i, trk_j, pmc_i, pmc_j, metxy_i):
        tar_ij = torch.cat([edge_index[0].view(-1, 1), trk_i], -1)
        tar_ji = torch.cat([edge_index[1].view(-1, 1), trk_j], -1)

        feats = []
        # 4-vector features - old state
        pmc_oij, pth_oi = graph_base.unique_aggregation(trk_i, pmc)
        pmc_oji, pth_oj = graph_base.unique_aggregation(trk_j, pmc)
        feats += addf(pmc_oij, pmc_oji) # 8

        # 4-vector features - new proposed state
        pmc_nij, pth_ni = graph_base.unique_aggregation(tar_ij, pmc)
        pmc_nji, pth_nj = graph_base.unique_aggregation(tar_ji, pmc)
        feats += addf(pmc_nij, pmc_nji) # 8

        # number of jumps old and new
        it_oi = (pth_oi > -1).sum(-1, keepdims = True)
        it_oj = (pth_oj > -1).sum(-1, keepdims = True)
        it_ni = (pth_ni > -1).sum(-1, keepdims = True)
        it_nj = (pth_nj > -1).sum(-1, keepdims = True)
        feats += addf(it_oi, it_oj) + addf(it_ni, it_nj) # 4

        # invariant masses of proposed edges
        mass_i  , mass_j   = physics.M(pmc_i)  , physics.M(pmc_j)
        mass_oij, mass_oji = physics.M(pmc_oij), physics.M(pmc_oji)
        mass_nij, mass_nji = physics.M(pmc_nij), physics.M(pmc_nji)
        feats += addf(mass_i, mass_j) + addf(mass_oij, mass_oji) + addf(mass_nij, mass_nji) # 6

        # delta R computation
        dr_ij  = physics.DeltaR(pmc_i, pmc_j)
        dr_oij = physics.DeltaR(pmc_oij, pmc_oji)
        dr_nij = physics.DeltaR(pmc_nij, pmc_nji)
        feats += addf(dr_ij, dr_oij) + addf(dr_ij, dr_nij) + addf(dr_oij, dr_nij) # 6

        # MET difference 
        met_ij , met_ji  = metxy_i -   pmc_i[:, :2], metxy_i -   pmc_j[:, :2]
        met_oij, met_oji = metxy_i - pmc_oij[:, :2], metxy_i - pmc_oji[:, :2]
        met_nij, met_nji = metxy_i - pmc_oij[:, :2], metxy_i - pmc_nji[:, :2]
        feats += addf(met_oij, met_oji) + addf(met_nij, met_nji) + addf(met_ij , met_ji) # 12
        feats = torch.cat(feats, -1)
        feats = torch.cat([feats, self._hidden[self._idx[edge_index[0], edge_index[1]]]], -1)
        return self.rnn_edge(feats.to(dtype = torch.float))

    def aggregate(self, message, edge_index, trk, pmc, metxy):
        src, dst = edge_index
        idx = self._idx[src, dst]
        self._tmp = self._hidden.clone()
        self._tmp[idx] = message
        self._hidden = self._tmp
        self._iter += 1

        # update the new path traversal
        sel = self.ret(message).max(-1)[1]
        try: gr = graph.edge(edge_index, sel, pmc, True)[1]
        except KeyError: return self._hidden

        edge_index_ = edge_index[:, sel != 1]
        if not edge_index_.size(1): return self._hidden
        trk = gr["clusters"][gr["reverse_clusters"]]
        return self.propagate(edge_index_, pmc = pmc, trk = trk, metxy = metxy)

    def forward(self, edge_index, batch, pmc, metxy):
        src, _ = edge_index
        nulls  = torch.zeros_like(batch).view(-1, 1)
        track  = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)

        self._hidden = torch.cat([nulls]*(self._in + self._repeat), -1)
        self._hidden = self._hidden[src].clone()
        self._hidden = self._hidden.to(dtype = torch.float)
        self._hidden = self.rnn_edge(self._hidden)

        self._idx = torch.cumsum(torch.ones_like(src), dim = -1)-1
        self._idx = to_dense_adj(edge_index, edge_attr = self._idx)[0]

        self._iter = 0
        out = self.propagate(edge_index, pmc = pmc, trk = track, metxy = metxy[batch])
        return self.ret(out)

class ExperimentalGNNv2(MessagePassing):

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
                Linear(34, self._n),
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

        out = self.rnn_edge(edge_index, batch, pmc, met_xy)
        try: gr = graph.edge(edge_index, out.max(-1)[1], pmc, True)[1]["node_sum"]
        except KeyError: gr = torch.zeros_like(pmc)

        data = torch.cat([pmc, gr - pmc, gr, pid, pmc[:, :2] - met_xy[batch]], -1)
        aggre = self.propagate(edge_index, tope = out, data = data)
        aggre = self._pool(aggre, batch)
        aggre = self.ntops(aggre)
        self.O_top_edge = out #self._drop(out)
        self.O_ntops = self._drop(aggre)
        self.iter = self.rnn_edge._iter

