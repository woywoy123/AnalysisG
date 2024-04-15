import torch
from torch import Tensor
from torch.nn import Sequential as Seq, Linear
from torch.nn import ReLU, Tanh, SELU, Dropout

from typing import Dict, List, Tuple

from torch_geometric.nn import MessagePassing, LayerNorm, aggr
from torch_geometric.utils import to_dense_adj
from pyc.interface import pyc_cuda

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

        self.O_top_edge: Tensor = torch.zeros((1))
        self.L_top_edge = "CEL"

        self.O_ntops: Tensor = torch.zeros((1))
        self.L_ntops = "CEL"

        self.soft_aggr = aggr.SoftmaxAggregation(learn = True)
        self.max_aggr  = aggr.MaxAggregation()
        self.var_aggr  = aggr.VarAggregation()

        # forward declaration
        self._h: Tensor = torch.zeros((1))
        self.pmu: Tensor = torch.zeros((1))
        self.pmc: Tensor = torch.zeros((1))
        self._cls: int = 0

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
                LayerNorm(self._hid),
                Linear(self._hid, self._hid),
                LayerNorm(self._hid),
                Linear(self._hid, self._rep)
        )
        self.rnn_x.apply(init_norm)

        self.rnn_mrg = Seq(
                Linear(self._rep*2, self._hid),
                LayerNorm(self._hid), ReLU(),
                Linear(self._hid, self._o)
        )
        self.rnn_mrg.apply(init_norm)

        self.node_feat  = Seq(
                Linear(18, self._rep),
                LayerNorm(self._rep),
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
                LayerNorm(self._hid),
                Linear(self._hid, 5)
        )
        self.graph_feat.apply(init_norm)

    def message(self, trk_i, trk_j, pmc_i, pmc_j):
        pmci: Tensor = pyc_cuda.graph.unique_aggregation(trk_i, self.pmc)[0]
        pmcj: Tensor = pyc_cuda.graph.unique_aggregation(trk_j, self.pmc)[0]
        pmc_ij, pth = pyc_cuda.graph.unique_aggregation(torch.cat([trk_i, trk_j], -1), self.pmc)

        m_i:  Tensor = pyc_cuda.combined.physics.cartesian.M(pmc_i)
        m_j:  Tensor = pyc_cuda.combined.physics.cartesian.M(pmc_j)
        m_ij: Tensor = pyc_cuda.combined.physics.cartesian.M(pmc_ij)
        dR:   Tensor = pyc_cuda.combined.physics.cartesian.DeltaR(pmci, pmcj)
        jmp:  Tensor = (pth > -1).sum(-1).view(-1,1)

        dx: List[Tensor] = [m_j, m_j - m_i, pmc_j, pmc_j - pmc_i]
        hdx: Tensor = self.rnn_dx(torch.cat(dx, -1).to(dtype = torch.float))

        _x: List[Tensor] = [m_ij, dR, jmp, pmc_ij]
        hx: Tensor = self.rnn_x(torch.cat(_x, -1).to(dtype = torch.float))
        return self.rnn_mrg(torch.cat([hx, hx - hdx], -1))

    def aggregate(self, message, edge_index, pmc, trk):
        return message

    def forward(self,
                edge_index: Tensor, batch: Tensor,
                G_met: Tensor, G_phi: Tensor, G_n_jets: Tensor, G_n_lep: Tensor,
                N_pT: Tensor, N_eta: Tensor, N_phi: Tensor, N_energy: Tensor,
                N_is_lep: Tensor, N_is_b: Tensor
        ) -> Tuple[Tensor, Tensor]:

        self.pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        self.pmc = pyc_cuda.combined.transform.PxPyPzE(self.pmu)

        batch:  Tensor = batch
        met_xy: Tensor = torch.cat([pyc_cuda.separate.transform.Px(G_met, G_phi), pyc_cuda.separate.transform.Py(G_met, G_phi)], -1)
        pid:    Tensor = torch.cat([N_is_lep, N_is_b], -1)

        if self._nuR:
            self.pmc = pyc_cuda.nusol.combinatorial(edge_index, batch, self.pmc, pid, met_xy, self._gev)["pmc"]
            self.pmu = pyc_cuda.combined.transform.PtEtaPhiE(self.pmc)
            N_pT[:]  = self.pmu[:, 0].view(-1, 1)
            N_eta[:] = self.pmu[:, 1].view(-1, 1)
            N_phi[:] = self.pmu[:, 2].view(-1, 1)
            N_energy[:] = self.pmu[:, 3].view(-1, 1)

        self._cls = 0
        pmc: Tensor = self.pmc
        trk_: Tensor = torch.ones_like(N_pT).cumsum(0)-1
        self._h  = self.propagate(edge_index, pmc = pmc, trk = trk_)

        idx_mlp: Tensor = torch.cumsum(torch.ones_like(edge_index[0]), dim = -1)-1
        idx_mlp = to_dense_adj(edge_index, edge_attr = idx_mlp)[0]
        edge_index_: Tensor = edge_index
        while True:
            if not edge_index_.size(1): break
            H: Tensor = self.propagate(edge_index_, pmc = pmc, trk = trk_)
            sel: Tensor = H.max(-1)[1]

            H_: Tensor = self._h[idx_mlp[edge_index_[0], edge_index_[1]]]
            H = H_ - H
            if not sel.sum(-1): break
            gr_: Dict[str, Tensor]  = pyc_cuda.graph.edge_aggregation(edge_index_, H, self.pmc)[1]
            edge_index_ = edge_index_[:, sel != 1]
            trk_ = gr_["clusters"][gr_["reverse_clusters"]]
            self._cls += 1

        self.O_top_edge = self._h
        gr_: Dict[str, Tensor] = pyc_cuda.graph.edge_aggregation(edge_index, self.O_top_edge, self.pmc)[1]

        masses = pyc_cuda.combined.physics.cartesian.M(gr_["node_sum"])
        mT = torch.ones_like(masses) * 172.62 * (1000 if not self._gev else 1)
        mW = torch.ones_like(masses) * 80.385 * (1000 if not self._gev else 1)
        mass_delta = torch.cat([mT - masses, mW - masses], -1)

        sft = self.soft_aggr(self.O_top_edge, edge_index[0])
        mx  = self.max_aggr(self.O_top_edge, edge_index[0])
        var = self.var_aggr(self.O_top_edge, edge_index[0])

        feat  = [gr_["node_sum"], pyc_cuda.combined.physics.cartesian.M(gr_["node_sum"])]
        feat += [self.pmc, pyc_cuda.combined.physics.cartesian.M(self.pmc)]
        feat += [pid, sft, mx, var]
        node = self.node_feat(torch.cat(feat, -1).to(dtype = torch.float))

        sft = self.soft_aggr(node, batch)
        mx  = self.max_aggr(node, batch)
        var = self.var_aggr(node, batch)

        feat  = [met_xy[batch] - gr_["node_sum"][:, :2]]
        feat += [met_xy[batch] - self.pmc[:, :2], mass_delta]
        node_dx = self.node_delta(torch.cat(feat, -1).to(dtype = torch.float))

        sft_dx = self.soft_aggr(node_dx, batch)
        mx_dx  = self.max_aggr(node_dx, batch)
        var_dx = self.var_aggr(node_dx, batch)
        self.O_ntops = self.graph_feat(torch.cat([sft, sft_dx, mx, mx_dx, var, var_dx], -1))

        return self.O_top_edge, self.O_ntops
