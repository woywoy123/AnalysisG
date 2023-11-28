import pyc
import pyc.Graph.Cartesian as graph
import pyc.Graph.Base as graph_base
import pyc.NuSol.Polar as nusol
import pyc.Physics.Cartesian as physics
import pyc.Transform as transform
import torch

def NuNuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk, idx):

    i, j = edge_index

    pmu = pmu.to(dtype = torch.double)
    pmu_i = torch.zeros_like(pmu[i])
    pmu_j = torch.zeros_like(pmu[j])
    sol_feat = torch.zeros_like(i).to(dtype = torch.double)-1
    if not msk.sum(-1): return sol_feat, pmu_i, pmu_j

    # Remove edges where the src/dst are neither bs or leps
    msk_ = (pid[i] + pid[j]).sum(-1) == 2


    # Only add keep edges where the src is the b-jet and the target is a lepton
    # and also consider the other direction
    msk_i = msk_ * (pid[i][:, 1]*pid[j][:, 0]) == 1
    msk_j = msk_ * (pid[j][:, 1]*pid[i][:, 0]) == 1

    # Remove batches which are non dilepton
    msk_ij = (msk_i + msk_j)#*msk

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

    print(l1.size())
    # Run the algorithm
    masses = masses.to(dtype = torch.double)
    met_phi = torch.cat([G_met[bt], G_phi[bt]], -1).to(dtype = torch.double)
    _sols = nusol.NuNu(b1, b2, l1, l2, met_phi, masses, 10e-8)
    nu1, nu2, dist, _, _, _, nosol = _sols
    print(dist[nosol == False].tolist()[0][0])

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


