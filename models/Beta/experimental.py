import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
from torch.nn import Dropout

import pyc.Transform as transform
import pyc.NuSol.Cartesian as nusol
import pyc.Graph.Base as graph_base
import pyc.Physics.Cartesian as physics


torch.set_printoptions(precision=3, sci_mode = True, linewidth = 1000)

def init_norm(m):
    if not type(m) == nn.Linear: return
    nn.init.uniform(m.weight)


class RecursiveParticles(MessagePassing):

    def __init__(self, disable_nu = True):
        super().__init__(aggr = None)
        end = 32
        self.output = None
        self._no_nu = disable_nu
        self.rnn_edge = Seq(Linear(2, end), Linear(end, 2))

    def nu(self, pmc, targets, msk_lep, msk_b, nu1, nu2):
        if self._no_nu: return
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
        if self._no_nu: return
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

    def message(self, pmc_i, pmc_j, path_p_j, trk_j, trk_i, pid):

        prop_path = torch.cat([path_p_j, trk_i], -1)
        is_lep, is_b = pid[:, 0], pid[:, 1]
        msk_lep = is_lep[prop_path]*(prop_path > -1)
        msk_b   = is_b[prop_path]*(prop_path > -1)

        nu1, nu2 = torch.zeros_like(pmc_i), torch.zeros_like(pmc_j)
        self.nu(self._pmc, prop_path, msk_lep, msk_b, nu1, nu2)
        self.nunu(self._pmc, prop_path, msk_lep, msk_b, nu1, nu2)

        # features
        pmc_ij, uniq = graph_base.unique_aggregation(prop_path, self._pmc)
        mass_ij = physics.M(pmc_ij + nu1 + nu2).to(dtype = torch.float)
        iters_ij = (uniq > -1).sum(-1, keepdim = True).to(dtype = torch.float)

        feats = [mass_ij, iters_ij]
        feats = torch.cat(feats, -1)

        edge_ = torch.cat([trk_j.view(1, -1), trk_i.view(1, -1)], 0)
        mlp = self.rnn_edge(feats)
        return mlp, mlp.softmax(-1), edge_

    def aggregate(self, message, path_p, pid):
        mlp, mlp_, edge_ = message
        n_nodes = self._pmc.size(0)

        if self._mlp is None: self._mlp = mlp
        adj_n = to_dense_adj(edge_, edge_attr = mlp_[:, 1], max_num_nodes = n_nodes)[0]
        val, idx = adj_n.max(-1)
        idx[val == 0] = -1
        if not val.sum(-1): return self._mlp
        pmc_, self._path = graph_base.unique_aggregation(torch.cat([self._path, idx.view(-1, 1)], -1), self._pmc)

        mass_  = physics.M(pmc_).to(torch.float)[edge_[0]]
        iters_ = (self._path[edge_[0]] > -1).sum(-1, keepdim = True).to(dtype = torch.float)
        feats_ = torch.cat([mass_, iters_], -1)
        mlp__  = self.rnn_edge(feats_)

        msk   = mlp_[:, 1] < mlp__.softmax(-1)[:, 1]
        msk__ = self._msk.clone()
        msk__[self._msk] = msk == False
        self._mlp[msk__] = mlp__[msk == False]
        self._idx[msk__] = -1
        self._msk = self._idx > -1

        if not edge_[:, msk].size(1): return self._mlp
        return self.propagate(edge_[:, msk], pmc = self._pmc, path_p = self._path, trk = self._track, pid = pid)

    def forward(self, edge_index, batch, pmc, pid, met_xy):
        self._pmc   = pmc
        self._mlp   = None

        mT = torch.ones_like(batch.view(-1, 1)) * 172.62 * 1000
        mW = torch.ones_like(batch.view(-1, 1)) * 80.385 * 1000
        mN = torch.ones_like(batch.view(-1, 1)) * 0

        self.met_xy = met_xy
        self.masses = torch.cat([mW, mT, mN], -1).to(dtype = torch.double)
        self.SXX = torch.tensor([[100, 0, 0, 100]], device = pmc.device, dtype = pmc.dtype)
        self.batch = batch

        self._idx   = torch.ones_like(edge_index[0]).cumsum(-1)-1
        self._path  = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)
        self._track = self._path.clone()
        self._msk   =  self._idx > -1

        self.output = self.propagate(edge_index, pmc = pmc, path_p = self._path, trk = self._track, pid = pid)

class ExperimentalGNN(MessagePassing):

    def __init__(self):
        super().__init__()
        self.rnn_edge = RecursiveParticles(True)
        self.rnn_edge.to(device = "cuda")
        self.O_top_edge = None
        self.L_top_edge = "CEL"

        self._drop = Dropout(p = 0.5)

    def forward(self, edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b):

        pid = torch.cat([N_is_lep, N_is_b], -1)
        met_x = transform.Px(G_met, G_phi)
        met_y = transform.Py(G_met, G_phi)
        met_xy = torch.cat([met_x, met_y], -1)

        pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)
        pmc = transform.PxPyPzE(pmu)

        self.rnn_edge(edge_index, batch, pmc, pid, met_xy)
        self.O_top_edge = self.rnn_edge.output
        self._l = self.rnn_edge._path
        self.O_top_edge = self._drop(self.O_top_edge)

