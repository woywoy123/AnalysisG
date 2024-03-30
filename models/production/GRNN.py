from torch_geometric.nn import MessagePassing, LayerNorm, aggr
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv

import torch
from torch.nn import Sequential as Seq, Linear
from torch.nn import ReLU

import pyc
import pyc.Transform as transform
import pyc.Graph.Base as graph
import pyc.Physics.Cartesian as physics
import pyc.NuSol.Cartesian as nusol


class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):

        super().__init__(aggr = None)

        try: dev = self.__param__["device"]
        except AttributeError: dev = "cuda:1"

        try: self._gev = self.__param__["gev"]
        except AttributeError: self._gev = False

        self.O_top_edge = None
        self.L_top_edge = "CEL"

        self._o = 2
        self._rep = 32

        self._dx  = 5
        self.rnn_dx = Seq(
                Linear(self._dx*2, self._rep),
                LayerNorm(self._rep), ReLU(),
                Linear(self._rep, self._rep)
        )

        self._x   = 7
        self.rnn_x = Seq(
                Linear(self._x, self._rep),
                LayerNorm(self._rep),
                Linear(self._rep, self._rep)
        )

        self.rnn_mrg = Seq(
                Linear(self._rep*2, self._rep),
                ReLU(),
                Linear(self._rep, self._o)
        )


    def message(self, trk_i, trk_j, pmc_i, pmc_j):
        pmci ,   _ = graph.unique_aggregation(trk_i, self.pmc)
        pmcj ,   _ = graph.unique_aggregation(trk_j, self.pmc)
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

        if cls >= self._cls: return self._hid
        self._cls = cls
        self.iter += 1

        return self.propagate(edge_index, pmc = gr_["node_sum"], trk = trk)

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

        #data = nusol.Combinatorial(edge_index, batch, self.pmc, self.pid, self.met_xy, null = 10e-10, gev = False)

        self.iter = 0
        self._hid = None
        self._cls = N_pT.size(0)
        self._t   = torch.ones_like(N_pT).cumsum(0)-1

        self.O_top_edge = self.propagate(edge_index, pmc = self.pmc, trk = self._t)


