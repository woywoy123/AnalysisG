from nu import NuNuCombinatorial, NuCombinatorial
from torch_geometric.utils import to_dense_adj
from AnalysisG.IO import UnpickleObject
from torch_geometric.data import Batch
import pyc.Transform as transform
import pyc.NuSol as nusol
import torch
torch.set_printoptions(precision=3, sci_mode = False)

x = UnpickleObject("data/GraphChildrenNoNu")
data = Batch().from_data_list(x[:2]).to(device = "cuda")
i, j = data.edge_index
bi = i[data.N_is_b[i].view(-1)   * data.N_is_lep[j].view(-1)]
lj = j[data.N_is_lep[j].view(-1) * data.N_is_b[i].view(-1)]

li = i[data.N_is_lep[i].view(-1) * data.N_is_b[j].view(-1)]
bj = j[data.N_is_b[j].view(-1)   * data.N_is_lep[i].view(-1)]

ij_b = torch.cat([bi, bj], -1)
ji_b = torch.cat([bj, bi], -1)

ij_l = torch.cat([li, lj], -1)
ji_l = torch.cat([lj, li], -1)

msk = (ij_b != ji_b)*(ij_l != ji_l)
#print(ij_b)
#print(ji_b)
#print(ij_l)
#print(ji_l)
#print("____")
#print(ij_b[msk])
#print(ji_b[msk])
#print(ij_l[msk])
#print(ji_l[msk])

N_is_b, N_is_lep = data.N_is_b, data.N_is_lep
N_pT, N_eta, N_phi, N_energy = data.N_pT, data.N_eta, data.N_phi, data.N_energy
G_met, G_phi = data.G_met, data.G_phi
pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], -1)

met = torch.cat([G_met, G_phi], -1)
one = torch.ones_like(met)
met_updown = torch.cat([one*0.2, one*2], -1)
mass_updown = torch.cat([one*0.2, one*2], -1)

mT = torch.ones_like(N_pT.view(-1, 1)) * 172.62*1000
mW = torch.ones_like(N_pT.view(-1, 1)) * 80.385*1000

mass_nom = torch.cat([mT, mW], -1)
pid = torch.cat([N_is_lep, N_is_b], -1)

for i in range(10):
    x = nusol.Polar.NuNuCombinatorial(data.edge_index, data.batch, pmu, pid, met, met_updown, mass_nom, mass_updown, step_size = 0.05, null = 1e-6)
    print(i)
exit()



N_pT, N_eta, N_phi, N_energy = data.N_pT, data.N_eta, data.N_phi, data.N_energy
N_is_b, N_is_lep = data.N_is_b, data.N_is_lep
edge_index, batch = data.edge_index, data.batch
G_met, G_phi = data.G_met, data.G_phi

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
_, n_lep = (batch[N_is_lep.view(-1) == 1]*1).unique(return_counts = True, dim = -1)

msk_nu   = ((n_bjets[batch] > 1)*(n_lep[batch] == 1))[src]
msk_nunu = ((n_bjets[batch] > 1)*(n_lep[batch] == 2))[src]

#chi2 = torch.zeros_like(src).to(dtype = torch.double)
#pmu_i, pmu_j = torch.zeros_like(pmu[src]), torch.zeros_like(pmu[src])

#chi2_n , pmu_i_n, pmu_j_n = NuCombinatorial  (edge_index, batch, pmu, pid, G_met, G_phi, masses, msk_nu)
#chi2_nn, pmu_inn, pmu_jnn = NuNuCombinatorial(edge_index, batch, pmu, pid, G_met, G_phi, masses, msk_nunu, idx)
#
#pmu_i[msk_nu]   += pmu_i_n[msk_nu]
#pmu_i[msk_nunu] += pmu_inn[msk_nunu]
#
#pmu_j[msk_nu]   += pmu_j_n[msk_nu]
#pmu_j[msk_nunu] += pmu_jnn[msk_nunu]
#
#chi2[msk_nu]   += chi2_n[msk_nu]
#chi2[msk_nunu] += chi2_nn[msk_nunu]


