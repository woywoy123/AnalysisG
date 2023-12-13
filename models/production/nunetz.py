import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, scatter
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh

# custom pyc extension functions
import pyc
import pyc.Physics.Cartesian as physics
import pyc.NuSol.Cartesian as nusol
import pyc.Graph.Cartesian as graph
import pyc.Graph.Base as graph_base
import pyc.Transform as transform

torch.set_printoptions(precision=3, sci_mode = True)

def init_norm(m):
    if not type(m) == nn.Linear: return
    nn.init.uniform(m.weight)

class ParticleRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None)
        self._in = 5
        self._out = 2
        self._rnn = 32

        _l = 128
        self._edge = Seq(Linear(self._in, _l), ReLU(), Tanh(), Linear(_l, self._rnn))
        self._red = Seq(Linear(self._rnn, self._out, bias = False))
        self._edge.apply(init_norm)

    def nu(self, pmc, targets, msk_lep, msk_b, nu1, nu2):
        msk = (msk_lep.sum(-1) == 1)*(msk_b.sum(-1) == 1)
        if msk.sum(-1) == 0: return
        l1 = pmc[targets[msk]][msk_lep[msk]]
        b1 = pmc[targets[msk]][msk_b[msk]]
        met_xy = self.met_xy[self.batch[targets[msk, 0]]]

        # create synthetic neutrinos
        nu, chi2 = nusol.Nu(b1, l1, met_xy, self.masses, self.SXX, 1e-8)
        chi_msk = ((chi2 != -1).cumsum(-1)-1) == 0
        nu_, chi2_ = nu[chi_msk], chi2[chi_msk]
        nu_ = torch.cat([nu_, nu_.pow(2).sum(-1, keepdim = True).pow(0.5)], -1)
        msk[msk.clone()] *= chi_msk.sum(-1) > 0

    def nunu(self, pmc, targets, msk_lep, msk_b, nu1, nu2):
        msk = (msk_lep.sum(-1) == 2)*(msk_b.sum(-1) == 2)
        if msk.sum(-1) == 0: return
        l1l2 = pmc[targets[msk]][msk_lep[msk]].view(-1, 8)
        b1b2 = pmc[targets[msk]][msk_b[msk]].view(-1, 8)
        l1, l2 = l1l2[:, :4].view(-1, 4), l1l2[:, 4:].view(-1, 4)
        b1, b2 = b1b2[:, :4].view(-1, 4), b1b2[:, 4:].view(-1, 4)
        met_xy = self.met_xy[self.batch[targets[msk, 0]]]
        l1, l2 = l1.contiguous(), l2.contiguous()
        b1, b2 = b1.contiguous(), b2.contiguous()

        # create synthetic neutrinos
        _sols = nusol.NuNu(b1, b2, l1, l2, met_xy, self.masses, 10e-8)
        nu1_, nu2_, dist, _, _, _, nosol = _sols
        if not dist.size(1): return
        is_sol = nosol == False

        nu1_, nu2_ = nu1_[:, 0, :][is_sol], nu2_[:, 0, :][is_sol]
        nu1_ = torch.cat([nu1_, (nu1_.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1)
        nu2_ = torch.cat([nu2_, (nu2_.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1)

        msk[msk.clone()] *= is_sol
        nu1[msk] = nu1_ + l1[is_sol] + b1[is_sol]
        nu2[msk] = nu2_ + l2[is_sol] + b2[is_sol]

    def message(self, edge_index, pmc, pmc_i, pmc_j, trk_i, trk_j, pid):
        src, dst = edge_index
        target = torch.cat([src.view(-1, 1), trk_i], -1)
        pmc_ij = graph_base.unique_aggregation(target, pmc)

        is_lep, is_b = pid[:, 0], pid[:, 1]
        msk_lep = is_lep[target]*(target > -1)
        msk_b   = is_b[target]*(target > -1)

        nu1, nu2 = torch.zeros_like(pmc_i), torch.zeros_like(pmc_j)
        self.nu(pmc, target, msk_lep, msk_b, nu1, nu2)
        self.nunu(pmc, target, msk_lep, msk_b, nu1, nu2)

        norm = (target > -1).sum(-1, keepdim = True)
        feats  = [physics.M(pmc_ij)      , physics.M(nu1)      ]
        feats += [physics.M(nu2), physics.M(pmc_ij + nu1 + nu2)]
        feats += [norm]

        feats = torch.cat(feats, -1)
        mlp = self._edge(feats.to(dtype = torch.float))
        return edge_index, nu1, nu2, mlp, self._red(mlp).max(-1)[1]

    def aggregate(self, message, trk, pmc, pid):
        edge_index, nu1, nu2, mlp, sel = message
        src, dst = edge_index
        idx = self._idx_mlp[dst, src]
        self._hidden[idx] = mlp

        try: gr = graph.edge(edge_index, sel, pmc, True)[1]
        except KeyError: return self._hidden

        trk = gr["clusters"][gr["reverse_clusters"]]
        edge_index_ = edge_index[:, sel != 1]
        if not edge_index_.size(1): return self._hidden

        return self.propagate(edge_index_, pmc = pmc, trk = trk, pid = pid)

    def forward(self, edge_index, batch, pmc, pid, met_xy):

        mass = physics.M(pmc)
        one = torch.ones_like(mass)
        feats = [mass, one, one, one, one]

        mT = torch.ones_like(mass) * 172.62
        mW = torch.ones_like(mass) * 80.385
        mN = torch.zeros_like(mW)

        self.met_xy = met_xy
        self.masses = torch.cat([mW, mT, mN], -1)
        self.SXX = torch.tensor([[100, 0, 0, 100]], device = pmc.device, dtype = pmc.dtype)
        self.batch = batch

        track = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)
        self._hidden = torch.cat(feats, dim = -1)
        self._hidden = self._hidden[edge_index[0]].clone()
        self._hidden = self._hidden.to(dtype = torch.float)
        self._hidden = self._edge(self._hidden)

        self._idx_mlp = torch.cumsum(torch.ones_like(edge_index[0]), dim = -1)-1
        self._idx_mlp = to_dense_adj(edge_index, edge_attr = self._idx_mlp)[0]

        return self._red(self.propagate(edge_index, pmc = pmc, trk = track, pid = pid))

class RecursiveNuNetz(MessagePassing):

    def __init__(self):
        super().__init__()
        self._top = ParticleRecursion()
        self.O_top_edge = None
        self.L_top_edge = "CEL"

        self.O_ntops = None
        self.L_ntops = "CEL"

        _l = 32
        self._in = 14
        self._ntops = Seq(Linear(self._in*2, _l), Tanh(), ReLU(), Linear(_l, 5))

    def message(self, feats_i, feats_j, istop):
        feat_i = torch.cat([feats_i, istop.view(-1, 1)], -1)
        feat_j = torch.cat([feats_i, istop.view(-1, 1)], -1)
        mlp = self._ntops(torch.cat([feat_i - feat_j, feat_i], -1).to(dtype = torch.float))
        return mlp

    def forward(self, edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b):

        pmu = torch.cat([N_pT / 1000, N_eta, N_phi, N_energy / 1000], -1)
        pmc = transform.PxPyPzE(pmu)
        pid = torch.cat([N_is_lep, N_is_b], -1)
        met_x = transform.Px(G_met, G_phi)
        met_y = transform.Py(G_met, G_phi)
        met_xy = torch.cat([met_x, met_y], -1)
        self.O_top_edge = self._top(edge_index, batch, pmc, pid, met_xy)

        _, tops = self.O_top_edge.max(-1)
        _graph = graph.edge(edge_index, tops, pmu, True)
        try: gr = _graph[1]
        except KeyError: gr = _graph[0]

        trk = gr["clusters"][gr["reverse_clusters"]]
        pmc_ = graph_base.unique_aggregation(trk, pmc)
        feats = [G_met[batch], G_phi[batch], pid[batch]]
        feats += [pmc_, pmc, (trk > -1).sum(-1, keepdim = True)]
        feats = torch.cat(feats, -1).clone()

        ntops = self.propagate(edge_index, feats = feats, istop = tops)
        self.O_ntops = scatter(ntops, batch, 0)


        exit()
