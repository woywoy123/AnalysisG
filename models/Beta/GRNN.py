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

torch.set_printoptions(precision=3, linewidth = 1000, threshold = 10000)

class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):

        super().__init__(aggr = None)

        try: dev = self.__param__["device"]
        except AttributeError: dev = "cuda"

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

        mT = torch.ones([1, 1]) * 172.62 * 1000
        mW = torch.ones([1, 1]) * 80.385 * 1000
        mN = torch.zeros([1, 1])
        self._masses = torch.cat([mW, mT, mN], -1).to(device = dev)
        self._SXX = torch.tensor([[100, 0, 0, 100]], device = dev, dtype = torch.double)


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

    def nu(self):
        is_lep, is_b, nlep = self.pid[:, 0], self.pid[:, 1], self.pid[:, 2]
        pair_msk  = is_lep[self.edge_idx[0]] * is_b[self.edge_idx[1]]
        pair_msk *= nlep[self.edge_idx[0]] == 1
        w = torch.zeros_like(self.pmc[self.edge_idx[0]])
        if pair_msk.sum(-1) == 0: return w
        pair_msk = pair_msk != 0
        l1 = self.pmc[self.edge_idx[0]][pair_msk]
        b1 = self.pmc[self.edge_idx[1]][pair_msk]
        met_xy = self.met_xy[self.batch[self.edge_idx[0]]]

        # create synthetic neutrinos
        nu, chi2 = nusol.Nu(b1, l1, met_xy, self.masses, self._SXX, 1e-8)
        chi_msk = ((chi2 != -1).cumsum(-1)-1) == 0
        nu_, chi2_ = nu[chi_msk], chi2[chi_msk]
        nu_ = torch.cat([nu_, nu_.pow(2).sum(-1, keepdim = True).pow(0.5)], -1)
        pair_msk[pair_msk.clone()] *= chi_msk.sum(-1) > 0

        w[pair_msk] += nu_ + b1
        return w

    def nunu(self):
        i, j = self.edge_idx
        is_l, is_b, nlep = self.pid[:, 0], self.pid[:, 1], self.pid[:, 2]
        pair_ll = (i != j)*(is_l[i] * is_l[j]) > 0
        pair_bb = (i != j)*(is_b[j] * is_b[i]) > 0
        pair_bl = (i != j)*(is_b[j] * is_l[i]) > 0
        pair_lb = (i != j)*(is_l[j] * is_b[i]) > 0
        pairs = pair_bb + pair_ll + pair_bl + pair_lb
        pairs *= nlep[i] == 2
        w1 = torch.zeros_like(self.pmc)[i]
        w2 = torch.zeros_like(self.pmc)[i]
        if not pairs.sum(-1): return

        l1l2 = torch.cat([i[pairs].view(-1, 1), j[pairs].view(-1, 1)], -1)
        b1b2 = torch.cat([i[pairs].view(-1, 1), j[pairs].view(-1, 1)], -1)

        dx = (1*pairs[pairs] > -1).cumsum(-1)-1
        dx_i = dx.view(1, -1)[(0*pairs[pairs])]
        dx_j = dx_i.transpose(0, 1)
        matrix = torch.cat([b1b2[dx_i.reshape(-1)], l1l2[dx_j.reshape(-1)]], -1)

        bx_i, bx_j, lx_i, lx_j = matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]
        msk = (is_b[bx_i] + is_b[bx_j] + is_l[lx_i] + is_l[lx_j]) == 4
        b1, b2, l1, l2 = self.pmc[bx_i[msk]], self.pmc[bx_j[msk]], self.pmc[lx_i[msk]], self.pmc[lx_j[msk]]
        l1, l2 = l1.contiguous(), l2.contiguous()
        b1, b2 = b1.contiguous(), b2.contiguous()
        met_xy = self.met_xy[bx_i[msk]]

        # create synthetic neutrinos
        _sols = nusol.NuNu(b1, b2, l1, l2, met_xy, self.masses, 10e-8)
        nu1_, nu2_, dist, _, _, _, nosol = _sols
        if not dist.size(1): return
        is_sol = nosol == False

        nu1_, nu2_ = nu1_[:, 0, :][is_sol], nu2_[:, 0, :][is_sol]
        tmp1 = torch.cat([nu1_, (nu1_.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1) + l1[is_sol]
        tmp2 = torch.cat([nu2_, (nu2_.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1) + l2[is_sol]

        w1[self._index[bx_i[msk][is_sol], lx_i[msk][is_sol]]] = tmp1

        w2[self._index[bx_j[msk][is_sol], lx_j[msk][is_sol]]] = tmp2
        return torch.cat([w1, w2], -1)

    def forward(self,
                edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep,
                N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b
        ):

        scale = 1/1000 if self._gev else 1
        self.pmu = torch.cat([N_pT*scale, N_eta, N_phi, N_energy*scale], -1)
        self.pmc = transform.PxPyPzE(self.pmu)
        self._index   = to_dense_adj(edge_index, edge_attr = (edge_index[0] > -1).cumsum(-1)-1)[0]

        self.batch    = batch
        self.edge_idx = edge_index
        self.pid      = torch.cat([N_is_lep, N_is_b, G_n_lep[batch]], -1)
        self.masses   = self._masses*torch.ones_like(N_pT)*scale
        self.met_xy   = scale*torch.cat([transform.Px(G_met, G_phi), transform.Py(G_met, G_phi)], -1)[batch]

        self.iter = 0
        self._hid = None
        self._cls = N_pT.size(0)
        self._t   = torch.ones_like(N_pT).cumsum(0)-1

        self.nu()
        self.nunu()
        self.O_top_edge = self.propagate(edge_index, pmc = self.pmc, trk = self._t)

