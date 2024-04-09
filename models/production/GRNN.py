from torch_geometric.nn import MessagePassing, LayerNorm, aggr
from torch_geometric.utils import to_dense_adj

import torch
from torch.nn import Sequential as Seq, Linear
from torch.nn import ReLU, Tanh, SELU, Dropout

import pyc
import pyc.Transform as transform
import pyc.Graph.Base as graph
import pyc.Physics.Cartesian as physics
import pyc.NuSol.Cartesian as nusol

def init_norm(m):
    if not type(m) == torch.nn.Linear: return
    torch.nn.init.uniform(m.weight, -1, 1)


class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):

        super().__init__(aggr = None)

        try: self._gev = self.__param__["gev"]
        except AttributeError: self._gev = False

        try: self._nuR = self.__params__["nu_reco"]
        except AttributeError: self._nuR = False
        self._nuR = False
        self.O_top_edge = None
        self.L_top_edge = "CEL"

        self.O_ntops = None
        self.L_ntops = "CEL"

        self.soft_aggr = aggr.SoftmaxAggregation(learn = True)
        self.max_aggr  = aggr.MaxAggregation()
        self.var_aggr  = aggr.VarAggregation()

        self._o = 2
        self._rep = 32
        self._hid = 256

        self._dx  = 5
        self.rnn_dx = Seq(
                Linear(self._dx*2, self._hid),
                LayerNorm(self._hid), ReLU(),
                Linear(self._hid, self._hid),
                LayerNorm(self._hid), ReLU(),
                Linear(self._hid, self._rep)
        )
        self.rnn_dx.apply(init_norm)

        self._x   = 7
        self.rnn_x = Seq(
                Linear(self._x, self._hid),
                LayerNorm(self._hid), Tanh(),
                Linear(self._hid, self._hid),
                LayerNorm(self._hid), Tanh(),
                Linear(self._hid, self._rep)
        )
        self.rnn_x.apply(init_norm)

        self.rnn_mrg = Seq(
                Linear(self._rep*2, self._hid),
                LayerNorm(self._hid), Tanh(),
                Linear(self._hid, self._o)
        )
        self.rnn_mrg.apply(init_norm)

        self.node_feat  = Seq(
                Linear(18, self._rep),
                LayerNorm(self._rep), Tanh(),
                Linear(self._rep, self._rep)
        )
        self.node_feat.apply(init_norm)

        self.node_delta = Seq(
                Linear(6, self._hid),
                LayerNorm(self._hid), ReLU(),
                Linear(self._hid, self._rep)
        )
        self.node_delta.apply(init_norm)

        self.graph_feat = Seq(
                Linear(self._rep*6, self._hid),
                LayerNorm(self._hid), Tanh(), ReLU(),
                Linear(self._hid, 5)
        )
        self.graph_feat.apply(init_norm)
        self._cache = {}

    def message(self, trk_i, trk_j, pmc_i, pmc_j):
        pmci ,    _ = graph.unique_aggregation(trk_i, self.pmc)
        pmcj ,    _ = graph.unique_aggregation(trk_j, self.pmc)
        pmc_ij, pth = graph.unique_aggregation(torch.cat([trk_i, trk_j], -1), self.pmc)

        m_i, m_j, m_ij = physics.M(pmci), physics.M(pmcj), physics.M(pmc_ij)
        jmp = (pth > -1).sum(-1, keepdims = True)
        dR  = physics.DeltaR(pmci, pmcj)

        dx = [m_j, m_j - m_i, pmc_j, pmc_j - pmc_i]
        self._hdx = self.rnn_dx(torch.cat(dx, -1).to(dtype = torch.float))

        _x = [m_ij, dR, jmp, pmc_ij]
        self._hx  = self.rnn_x(torch.cat(_x, -1).to(dtype = torch.float))

        self._hid = self.rnn_mrg(torch.cat([self._hx, self._hx - self._hdx], -1))
        return self._hid

    def aggregate(self, message, edge_index, pmc, trk):
        gr_  = graph.edge_aggregation(edge_index, message, self.pmc)[1]
        trk_ = gr_["clusters"][gr_["reverse_clusters"]]
        cls  = gr_["clusters"].size(0)

        msk = (trk_ > -1).sum(-1).max(-1)[1]
        trk_ = trk_[:, :msk]

        # enable this for max net
        #msg = to_dense_adj(edge_index, edge_attr = message.softmax(-1)[:, 1])[0].softmax(-1)
        #next_ = msg.max(-1)[1].view(-1, 1)
        #trk_ = torch.cat([trk, next_], -1)

        if not trk_.size(1): return self._hid
        if cls >= self._cls: return self._hid
        self._cls = cls
        self.iter += 1

        return self.propagate(edge_index, pmc = gr_["node_sum"], trk = trk_)

    def forward(self,
                edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep,
                N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b
        ):

        self.pmu      = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        self.pmc      = transform.PxPyPzE(self.pmu)
        self._index   = to_dense_adj(edge_index, edge_attr = (edge_index[0] > -1).cumsum(-1)-1)[0]

        self.batch    = batch
        self.edge_idx = edge_index
        self.pid      = torch.cat([N_is_lep, N_is_b], -1)
        self.met_xy   = torch.cat([transform.Px(G_met, G_phi), transform.Py(G_met, G_phi)], -1)

        t = self.pmu.sum(-1).sum(-1)/1000
        t = str(t.tolist())
        if self._nuR and t not in self._cache:
            data = nusol.Combinatorial(
                    edge_index, batch, self.pmc, self.pid, self.met_xy,
                    null = 10e-10, gev = self._gev, top_up_down = 0.95, w_up_down = 0.95
            )
            nu1, nu2, m1, m2, combi = [data[x] for x in ["nu_1f", "nu_2f", "ms_1f", "ms_2f", "combi"]]
            comb = combi.sum(-1) > 0
            l1, l2 = combi[comb, 2].to(dtype = torch.int64), combi[comb, 3].to(dtype = torch.int64)
            self.pmc[l1] += nu1[comb]
            self.pmc[l2] += nu2[comb]
            self.pmu = transform.PtEtaPhiE(self.pmc)
            self._cache[t] = self.pmu

        if t in self._cache:
            self.pmu = self._cache[t]
            N_pT[:] = self.pmu[:, 0].view(-1, 1)
            N_eta[:] = self.pmu[:, 1].view(-1, 1)
            N_phi[:] = self.pmu[:, 2].view(-1, 1)
            N_energy[:] = self.pmu[:, 3].view(-1, 1)
            self.pmc = transform.PxPyPzE(self.pmu)

        self.iter = 0
        self._hid = None
        self._cls = N_pT.size(0)
        self._t   = torch.ones_like(N_pT).cumsum(0)-1

        self.O_top_edge = self.propagate(edge_index, pmc = self.pmc, trk = self._t)

        gr_  = graph.edge_aggregation(edge_index, self.O_top_edge, self.pmc)[1]

        masses = physics.M(gr_["node_sum"])
        mT = torch.ones_like(masses) * 172.62 * (1000 if not self._gev else 1)
        mW = torch.ones_like(masses) * 80.385 * (1000 if not self._gev else 1)
        mass_delta = torch.cat([mT - masses, mW - masses], -1)

        sft = self.soft_aggr(self.O_top_edge, edge_index[0])
        mx  = self.max_aggr(self.O_top_edge, edge_index[0])
        var = self.var_aggr(self.O_top_edge, edge_index[0])

        feat  = [gr_["node_sum"], physics.M(gr_["node_sum"])]
        feat += [self.pmc, physics.M(self.pmc)]
        feat += [self.pid, sft, mx, var]
        node = self.node_feat(torch.cat(feat, -1).to(dtype = torch.float))

        sft = self.soft_aggr(node, batch)
        mx  = self.max_aggr(node, batch)
        var = self.var_aggr(node, batch)


        feat  = [self.met_xy[batch] - gr_["node_sum"][:, :2]]
        feat += [self.met_xy[batch] - self.pmc[:, :2], mass_delta]
        node_dx = self.node_delta(torch.cat(feat, -1).to(dtype = torch.float))

        sft_dx = self.soft_aggr(node_dx, batch)
        mx_dx  = self.max_aggr(node_dx, batch)
        var_dx = self.var_aggr(node_dx, batch)
        self.O_ntops = self.graph_feat(torch.cat([sft, sft_dx, mx, mx_dx, var, var_dx], -1))
