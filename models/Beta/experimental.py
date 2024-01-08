from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
from torch.nn import Dropout
import torch

import pyc.Transform as transform
import pyc.NuSol.Cartesian as nusol
import pyc.Graph.Base as graph_base
import pyc.Physics.Cartesian as physics
torch.set_printoptions(precision=3, sci_mode = True, linewidth = 1000)

def init_norm(m):
    if not type(m) == torch.nn.Linear: return
    torch.nn.init.uniform(m.weight)


class RecursiveParticles(MessagePassing):

    def __init__(self, disable_nu = True, gev = False):
        super().__init__(aggr = None)
        end = 128
        self._gev = gev
        self.output = None
        self._no_nu = disable_nu
        self.rnn_edge = Seq(Linear(2, end), Tanh(), ReLU(), Linear(end, 2))
        self.rnn_edge.apply(init_norm)

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

    def message(self, edge_index, pmc_i, pmc_j, trk_i, pid):

        self._path = torch.cat([self._path, trk_i], -1)
        pmc_ij, self._path = graph_base.unique_aggregation(self._path, self._pmc)

        is_lep, is_b = pid[:, 0], pid[:, 1]
        msk_lep = is_lep[self._path]*(self._path > -1)
        msk_b   = is_b[self._path]*(self._path > -1)

        nu1, nu2 = torch.zeros_like(pmc_i), torch.zeros_like(pmc_j)
        self.nu(self._pmc, self._path, msk_lep, msk_b, nu1, nu2)
        self.nunu(self._pmc, self._path, msk_lep, msk_b, nu1, nu2)

        # features
        mass_ij = physics.M(pmc_ij + nu1 + nu2).to(dtype = torch.float)
        iters_ij = (self._path > -1).sum(-1, keepdim = True).to(dtype = torch.float)

        feats = [mass_ij, iters_ij]
        feats = torch.cat(feats, -1)
        return edge_index, self.rnn_edge(feats)

    def aggregate(self, message, pid):
        edge_index, mlp = message
        msk = mlp.max(-1)[1] > 0
        self._mlp = mlp
        self._msk *= msk
        self._msk *= (self._path > -1).sum(-1) < self._num_nodes[edge_index[0]]
        self._iter += 1
        msk_it = self._num_nodes > self._iter
        if not msk_it.sum(-1): return self._mlp
        if not self._msk.sum(-1): return self._mlp
        return self.propagate(edge_index, pmc = self._pmc, trk = self._track, pid = pid)

    def forward(self, edge_index, batch, pmc, pid, met_xy):
        self._mlp   = torch.zeros_like(edge_index.view(-1, 2)).to(dtype = torch.float)
        self._pmc   = pmc
        src, _      = edge_index
        self._iter = 0

        if self._gev: f = 1
        else: f = 1000

        mT = torch.ones_like(batch.view(-1, 1)) * 172.62*f
        mW = torch.ones_like(batch.view(-1, 1)) * 80.385*f
        mN = torch.ones_like(batch.view(-1, 1)) * 0

        self.met_xy = met_xy
        self.masses = torch.cat([mW, mT, mN], -1).to(dtype = torch.double)
        self.SXX = torch.tensor([[100, 0, 0, 100]], device = pmc.device, dtype = pmc.dtype)
        self.batch = batch

        self._num_nodes = to_dense_adj(edge_index)[0].sum(-1)
        self._track = (torch.ones_like(batch).cumsum(-1)-1)
        self._path = self._track[src].clone().view(-1, 1)
        self._track = self._track.view(-1, 1)
        self._msk = src > -1

        self.output = self.propagate(edge_index, pmc = pmc, trk = self._track, pid = pid)

class ExperimentalGNN(MessagePassing):

    def __init__(self):
        super().__init__()
        self._gev = False
        self.rnn_edge = RecursiveParticles(True, False)
        self.rnn_edge.to(device = "cuda:1")
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
        self._l = self.rnn_edge._path
        self.O_top_edge = self._drop(self.O_top_edge)

