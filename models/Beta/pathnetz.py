import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, degree
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh

# custom pyc extension functions
import pyc
import pyc.Graph.Cartesian as graph
import pyc.Graph.Base as graph_base
import pyc.NuSol.Polar as nusol
import pyc.Physics.Cartesian as physics
import pyc.Transform as transform

torch.set_printoptions(precision=3, sci_mode = True)


def init_norm(m):
    if not type(m) == nn.Linear: return
    nn.init.uniform(m.weight)

def NuNuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk, idx, truth):
    i, j = edge_index
    pmu = pmu.to(dtype = torch.double)
    masses = masses.to(dtype = torch.double)
    sol_feat = torch.zeros_like(i).to(dtype = torch.double)-1
    pmu_i, pmu_j = torch.zeros_like(pmu[i]), torch.zeros_like(pmu[j])
    if not msk.sum(-1): return sol_feat, pmu_i, pmu_j

    # Remove edges where the src/dst are neither bs or leps
    msk_ = (pid[i] + pid[j]).sum(-1) == 2


    # Only add keep edges where the src is the b-jet and the target is a lepton
    # and also consider the other direction
    msk_i = msk_ * (pid[i][:, 1]*pid[j][:, 0]) == 1
    msk_j = msk_ * (pid[j][:, 1]*pid[i][:, 0]) == 1

    # Remove batches which are non dilepton
    msk_ij = (msk_i + msk_j)*msk

    # Find the original particle index in the event
    src, dst = edge_index[:, msk_ij]

    # create proxy particle indices (these are used to assign NON-TOPOLOGICALLY CONNECTED PARTICLE PAIRS)
    # e.g. 1 -> 2, 3 -> 4 is ok, but 1 -> 2, 1 -> 4 is not ok (they share the same lepton/b-quark).
    # This means NuNu(p1, p1, p2, p4) would be incorrect, we want NuNu(p1, p3, p2, p4)
    l1_, l2_ = src[pid[src, 0] == 1], dst[pid[dst, 0] == 1]
    b1_, b2_ = src[pid[src, 1] == 1], dst[pid[dst, 1] == 1]
    msk__ = (l1_ != l2_)*(b1_ != b2_)

    l1_, l2_ = l1_[msk__], l2_[msk__]
    b1_, b2_ = b1_[msk__], b2_[msk__]
    bt = batch[l1_]

    # Retrieve the associated four vectors 
    l1, l2 = pmu[l1_], pmu[l2_]
    b1, b2 = pmu[b1_], pmu[b2_]

    # Run the algorithm
    met_phi = torch.cat([G_met[bt], G_phi[bt]], -1).to(dtype = torch.double)
    _sols = nusol.NuNu(b1, b2, l1, l2, met_phi, masses, 10e-8)
    nu1, nu2, dist, _, _, _, nosol = _sols

    # Create a correct edge mapping
    if not dist.size(1): return sol_feat, pmu_i, pmu_j
    is_sol = nosol == False

    idx_ij1, idx_ji1 = idx[l1_, b1_], idx[b1_, l1_]
    idx_ij2, idx_ji2 = idx[l2_, b2_], idx[b2_, l2_]

    # Populate the null tensors with synthetic neutrinos 
    nu1 = torch.cat([nu1, (nu1.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1)
    nu2 = torch.cat([nu2, (nu2.pow(2).sum(-1, keepdim = True)).pow(0.5)], -1)

    sol_feat[idx_ij1[is_sol]] = dist[:, 0][is_sol]
    sol_feat[idx_ji1[is_sol]] = dist[:, 0][is_sol]

    sol_feat[idx_ij2[is_sol]] = dist[:, 0][is_sol]
    sol_feat[idx_ji2[is_sol]] = dist[:, 0][is_sol]

    pmu_i[idx_ji1[is_sol]] = nu1[:, 0, :][is_sol]
    pmu_j[idx_ij1[is_sol]] = nu1[:, 0, :][is_sol]

    pmu_i[idx_ji2[is_sol]] = nu2[:, 0, :][is_sol]
    pmu_j[idx_ij2[is_sol]] = nu2[:, 0, :][is_sol]

    return sol_feat, pmu_i, pmu_j

def NuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk):
    SXX = torch.tensor([[100, 0, 0, 100]], device = msk.device, dtype = pmu.dtype)
    i, j = edge_index

    # Output
    pmu_i = torch.zeros_like(pmu[i])
    pmu_j = torch.zeros_like(pmu[i])
    chi2_f = torch.zeros_like(i).to(dtype = torch.double)-1

    # make the network fully connected, and get only edges where src and dst are 
    # paired with leptons and b-jets
    pairs = (pid[i, 0] * pid[j, 1])*msk

    # find which one of the nodes is the lepton and the b-jet
    l_, b_ = i[pid[i, 0]*pairs], j[pid[j, 1]*pairs]
    if not msk.sum(-1): return chi2_f, pmu_i, pmu_j

    # Run the algorithm
    met_phi = torch.cat([G_met, G_phi], -1)[batch[l_]]
    nu, chi2 = nusol.Nu(pmu[b_], pmu[l_], met_phi, masses, SXX, 10e-8)

    # Create a mask such that 0 valued solutions are excluded
    nu_low_msk = ((chi2 != -1).cumsum(-1)-1) == 0
    nu_, chi2_ = nu[nu_low_msk], chi2[nu_low_msk]
    nu_ = torch.cat([nu_, nu_.pow(2).sum(-1, keepdim = True)], -1)

    nu_feat = torch.cat([nu_low_msk.sum(-1, keepdim = True)]*4, -1).to(dtype = torch.double)
    nu_feat[nu_feat.sum(-1) != 0] *= nu_

    chi2_feat = torch.cat([nu_low_msk.sum(-1, keepdim = True)], -1)
    chi2_feat = chi2_feat.to(dtype = torch.double).view(-1)
    chi2_feat[nu_feat.sum(-1) != 0] = chi2_

    # populate the nodes
    pmu_i[pairs] += nu_feat
    chi2_f[pairs] = chi2_feat

    return chi2_f, pmu_i, pmu_j



class ParticleRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None)
        self._in = 2
        self._out = 2

        self._edge = Seq(Linear(self._in*2, 128), Tanh(), Linear(128, self._in))
        self._red = Seq(Linear(self._in, self._out))

        self._edge.apply(init_norm)
        self._red.apply(init_norm)

    def message(self, edge_index, pmc, pmc_i, pmc_j, trk_i, trk_j, nu, nu_, sols):
        src, dst = edge_index
        idx = self._idx_mlp[src, dst]
        target = torch.cat([src.view(-1, 1), trk_i], -1)
        pmc_ij = graph_base.unique_aggregation(target, pmc) + nu[idx] + nu_[idx]

        feats = []
        feats += [physics.M(pmc_ij), physics.M(pmc_i + nu[idx] + nu_[idx])]
        feats += [self._hidden[idx]]

        feats = torch.cat(feats, -1)
        mlp = self._edge(feats.to(dtype = torch.float))
        return edge_index, mlp, self._red(mlp).max(-1)[1]

    def aggregate(self, message, trk, pmc, nu, nu_, sols):
        edge_index, mlp, sel = message
        src, dst = edge_index
        idx = self._idx_mlp[dst, src]

        try: gr = graph.edge(edge_index, sel, pmc, True)[1]
        except KeyError: return self._hidden
        except RuntimeError: return self._hidden

        trk = gr["clusters"][gr["reverse_clusters"]]
        feats = [physics.M(gr["node_sum"])]
        feats += [physics.M(pmc)]

        mlp = torch.cat([torch.cat(feats, -1)[src],  mlp], -1)
        self._hidden[idx] = self._edge(mlp.to(dtype = torch.float))
        edge_index_ = edge_index[:, sel != 1]
        return self.propagate(edge_index_, pmc = pmc, trk = trk, nu = nu, nu_ = nu_, sols = sols)

    def forward(self, edge_index, batch, pmc, nu_i, nu_j, sol):

        feats = [physics.M(pmc)]

        track = (torch.ones_like(batch).cumsum(-1)-1).view(-1, 1)
        self._hidden = torch.cat([track]*(self._in*2 - len(feats)), -1)
        self._hidden = feats + [torch.zeros_like(self._hidden)]

        self._hidden = torch.cat(self._hidden, -1)
        self._hidden = self._hidden[edge_index[0]].clone()
        self._hidden = self._hidden.to(dtype = torch.float)
        self._hidden = self._edge(self._hidden)

        self._idx_mlp = torch.cumsum(torch.ones_like(edge_index[0]), dim = -1)-1
        self._idx_mlp = to_dense_adj(edge_index, edge_attr = self._idx_mlp)[0]

        return self._red(self.propagate(edge_index, pmc = pmc, trk = track, nu = nu_i, nu_ = nu_j, sols = sol))

class RecursivePathNetz(MessagePassing):

    def __init__(self):
        super().__init__()
        self._top = ParticleRecursion()
        self.O_top_edge = None
        self.L_top_edge = "CEL"

    def NuNu(self, edge_index, batch, pmu, pid, G_met, G_phi, masses, msk, idx, truth):
        return NuNuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk, idx, truth)

    def Nu(self, edge_index, batch, pmu, pid, G_met, G_phi, masses, msk):
        return NuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk)

    def forward(self, edge_index, batch, G_met, G_phi, G_n_jets, G_n_lep, N_pT, N_eta, N_phi, N_energy, N_is_lep, N_is_b):

        pmu = torch.cat([N_pT / 1000, N_eta, N_phi, N_energy / 1000], -1)
        pmc = transform.PxPyPzE(pmu)
        pid = torch.cat([N_is_lep, N_is_b], -1)
        src, dst = edge_index

        mT = torch.ones_like(N_pT.view(-1, 1)) * 172.62
        mW = torch.ones_like(N_pT.view(-1, 1)) * 80.385
        mN = torch.zeros_like(mW)
        masses = torch.cat([mW, mT, mN], -1)

        idx = torch.cumsum(torch.ones_like(edge_index[0]), dim = -1)-1
        idx = to_dense_adj(edge_index, edge_attr = idx)[0]

        _, n_bjets = (batch[N_is_b.view(-1) == 1]*1).unique(return_counts = True, dim = -1)
        msk_nu   = ((n_bjets[batch] > 1)*(G_n_lep.view(-1)[batch] == 1))[src]
        msk_nunu = ((n_bjets[batch] > 1)*(G_n_lep.view(-1)[batch] == 2))[src]

        chi2 = torch.zeros_like(src).to(dtype = torch.double)
        pmu_i, pmu_j = torch.zeros_like(pmu[src]), torch.zeros_like(pmu[src])

        chi2_n , pmu_i_n, pmu_j_n = self.Nu(  edge_index, batch, pmu, pid, G_met, G_phi, masses, msk_nu)
        chi2_nn, pmu_inn, pmu_jnn = self.NuNu(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk_nunu, idx)

        pmu_i[msk_nu]   += pmu_i_n[msk_nu]
        pmu_i[msk_nunu] += pmu_inn[msk_nunu]

        pmu_j[msk_nu]   += pmu_j_n[msk_nu]
        pmu_j[msk_nunu] += pmu_jnn[msk_nunu]

        chi2[msk_nu]   += chi2_n[msk_nu]
        chi2[msk_nunu] += chi2_nn[msk_nunu]

        self.O_top_edge = self._top(edge_index, batch, pmc, pmu_i, pmu_j, chi2)
