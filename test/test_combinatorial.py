from AnalysisG import Analysis
from AnalysisG.Events import Event, GraphChildrenNoNu

def NuNuCombinatorial(self, edge_index, batch, Pmu, PiD, G_met, G_phi):
    i, j, batch = edge_index[0], edge_index[1], batch.view(-1)

    # Find edges where the source/dest are a b and lep
    msk = (((PiD[i] + PiD[j]) == 1).sum(-1) == 2) == 1

    # Block out nodes which are neither leptons or b-jets
    _i, _j = i[msk], j[msk]

    # Find the pairs where source particle is the lepton and the destination is a b-jet
    this_lep_i = (PiD[_i][:, 0] == 1).view(-1, 1)
    this_b_i = (PiD[_j][:, 1] == 1).view(-1, 1)
    p_msk_i = torch.cat([this_lep_i, this_b_i], -1).sum(-1) == 2  # enforce this

    # Find the pairs where destination particle is the lepton and the source is a b-jet
    this_lep_j = (PiD[_j][:, 0] == 1).view(-1, 1)
    this_b_j = (PiD[_i][:, 1] == 1).view(-1, 1)
    p_msk_j = torch.cat([this_lep_j, this_b_j], -1).sum(-1) == 2

    # Make sure the that source == destination (destination == source) particle index
    msk_ij = edge_index[:, msk][:, p_msk_i][0] == edge_index[:, msk][:, p_msk_j][1]
    msk_ji = edge_index[:, msk][:, p_msk_i][1] == edge_index[:, msk][:, p_msk_j][0]
    msk_ = msk_ji * msk_ij  # eliminates non-overlapping cases

    # Find the original particle index in the event
    par_ij = edge_index[:, msk][:, p_msk_i][:, msk_]

    # create proxy particle indices (these are used to assign NON-TOPOLOGICALLY CONNECTED PARTICLE PAIRS)
    # e.g. 1 -> 2, 3 -> 4 is ok, but 1 -> 2, 1 -> 4 is not ok (they share the same lepton/b-quark).
    # This means NuNu(p1, p1, p2, p4) would be incorrect, we want NuNu(p1, p3, p2, p4)
    nodes = par_ij.size()[1]
    dst = torch.tensor(
        [i for i in torch.arange(nodes)], dtype=torch.int, device=par_ij.device
    ).view(1, -1)
    src = torch.cat(
        [torch.ones_like(dst) * i for i in torch.arange(nodes)], -1
    ).view(-1)
    dst = torch.cat([dst for _ in torch.arange(nodes)], -1).view(-1)

    # Check whether the particles involved for these proxy node pairs are from the same event (batch).
    b_i = batch.view(-1)[par_ij[0][src]].view(-1)
    b_j = batch.view(-1)[par_ij[1][dst]].view(-1)

    # Make sure we dont double count. We do want cases where [p1, p3, p2, p4] <=> [p3, p1, p4, p2]
    # But not [p1, p1, p2, p4] <=> [p1, p1, p4, p2]
    b_ = (b_j == b_i) * (src != dst)

    # Get the original particle index of the b-jet and lepton for each event
    NuNu_i = par_ij[:, src[b_]]
    NuNu_j = par_ij[:, dst[b_]]

    # Make it look nicer
    NuNu_ = torch.cat([NuNu_i.t(), NuNu_j.t()], -1)

    b1, b2 = NuNu_[:, 1], NuNu_[:, 3]
    l1, l2 = NuNu_[:, 0], NuNu_[:, 2]






    #mT = torch.ones_like(b1.view(-1, 1)) * 172.62 * 1000
    #mW = torch.ones_like(b1.view(-1, 1)) * 80.385 * 1000
    #mN = torch.zeros_like(mW)
    #met, phi = G_met[batch[b1]], G_phi[batch[b1]]

    #_sols = NuSol.NuNuPtEtaPhiE(
    #    Pmu[b1], Pmu[b2], Pmu[l1], Pmu[l2], met, phi, mT, mW, mN, 10e-8
    #)

    #if len(_sols) == 5:
    #    return False
    #SkipEvent = _sols[0]

    ## Get the Neutrino Solutions
    #Pmc_nu1, Pmc_nu2 = _sols[1], _sols[2]

    ## Create a mask such that 0 valued solutions are excluded
    #nu1_msk = Pmc_nu1.sum(-1, keepdim = True) != 0
    #nu2_msk = Pmc_nu2.sum(-1, keepdim = True) != 0
    #_msk = (nu1_msk*nu2_msk).view(-1, 6)

    ## Compute the Neutrino 4-vector 
    #_e1, _e2 = Pmc_nu1.pow(2), Pmc_nu2.pow(2)
    #_e1, _e2 = _e1.sum(-1, keepdim = True), _e2.sum(-1, keepdim = True)
    #_e1, _e2 = _e1.pow(0.5), _e2.pow(0.5)
    #Pmc_nu1, Pmc_nu2 = torch.cat([Pmc_nu1, _e1], -1), torch.cat([Pmc_nu2, _e2], -1)

    ## Remove l which are marked as a skip event.
    #l1, l2 = l1[SkipEvent == False], l2[SkipEvent == False]

    ## Further Remove pairs based on null neutrino solutions 
    #l1, l2 = (l1.view(-1, 1)*_msk)[_msk], (l2.view(-1, 1)*_msk)[_msk]

    ## Collect only the leptons from the original vector and match them to the neutrino
    #lep1 = Tr.PxPyPz(Pmu[l1][:, 0].view(-1, 1), Pmu[l1][:, 1].view(-1, 1), Pmu[l1][:, 2].view(-1, 1))
    #lep2 = Tr.PxPyPz(Pmu[l2][:, 0].view(-1, 1), Pmu[l2][:, 1].view(-1, 1), Pmu[l2][:, 2].view(-1, 1))

    #W1 = torch.cat([lep1, Pmu[l1][:, 3].view(-1, 1)], -1) + Pmc_nu1[_msk]
    #W2 = torch.cat([lep2, Pmu[l2][:, 3].view(-1, 1)], -1) + Pmc_nu2[_msk]

    #Pmu_W1 = torch.cat([Tr.PtEtaPhi(W1[:, 0].view(-1, 1), W1[:, 1].view(-1, 1), W1[:, 2].view(-1, 1)), W1[:, 3].view(-1, 1)], -1)
    #Pmu_W2 = torch.cat([Tr.PtEtaPhi(W2[:, 0].view(-1, 1), W2[:, 1].view(-1, 1), W2[:, 2].view(-1, 1)), W2[:, 3].view(-1, 1)], -1)

    #return Pmu_W1, Pmu_W2, W1, W2, l1, l2


