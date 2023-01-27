from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event 
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
import ROOT as r
import torch
import NuR.DoubleNu.Floats as Sf
import NuR.Physics.Floats as F
from neutrino_momentum_reconstruction_python3 import doubleNeutrinoSolutions

mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass
device = "cuda"

def CompareNumerical(r_ori, r_pyt, string):
    print("(" + string + ") -> Original: ", r_ori, " ||  Pytorch: ", r_pyt, " || Error (%): ", 100*abs(r_pyt - r_ori)/r_ori)

def _MakeTensor(val, n, device = "cpu", dtp = torch.double):
    return torch.tensor([val for i in range(n)], device = device, dtype = dtp)

def DoubleNeutrino(b, mu, ev):
    r_b1 = r.TLorentzVector()
    r_b1.SetPtEtaPhiE(b[0].pt, b[0].eta, b[0].phi, b[0].e)
    
    r_mu1 = r.TLorentzVector()
    r_mu1.SetPtEtaPhiE(mu[0].pt, mu[0].eta, mu[0].phi, mu[0].e)

    r_b2 = r.TLorentzVector()
    r_b2.SetPtEtaPhiE(b[1].pt, b[1].eta, b[1].phi, b[1].e)
    
    r_mu2 = r.TLorentzVector()
    r_mu2.SetPtEtaPhiE(mu[1].pt, mu[1].eta, mu[1].phi, mu[1].e)

    x = F.ToPx(ev.met, ev.met_phi, "cpu").tolist()[0][0]
    y = F.ToPy(ev.met, ev.met_phi, "cpu").tolist()[0][0]

    return doubleNeutrinoSolutions(r_b1, r_b2, r_mu1, r_mu2, x, y)
    
def DoubleNeutrinoPyT(b, mu, ev):

    _b1 = torch.tensor([[b_[0].pt, b_[0].eta, b_[0].phi, b_[0].e] for b_ in b], dtype = torch.double, device = device)
    _b2 = torch.tensor([[b_[1].pt, b_[1].eta, b_[1].phi, b_[1].e] for b_ in b], dtype = torch.double, device = device)

    _mu1 = torch.tensor([[mu_[0].pt, mu_[0].eta, mu_[0].phi, mu_[0].e] for mu_ in mu], dtype = torch.double, device = device)
    _mu2 = torch.tensor([[mu_[1].pt, mu_[1].eta, mu_[1].phi, mu_[1].e] for mu_ in mu], dtype = torch.double, device = device)

    _met = torch.tensor([[ev_.met] for ev_ in ev], dtype = torch.double, device = device)
    _phi = torch.tensor([[ev_.met_phi] for ev_ in ev], dtype = torch.double, device = device)
    
    n = len(b)
    _mT = _MakeTensor([mT], n, device)
    _mW = _MakeTensor([mW], n, device)
    _mN = _MakeTensor([mN], n, device)
   
    return Sf.SolT(_b1, _b2, _mu1, _mu2, _mT, _mW, _mN, _met, _phi)

def ParticleCollectors(ev):
    t1 = [ t for t in ev.Tops if t.DecayLeptonically()][0]
    t2 = [ t for t in ev.Tops if t.DecayLeptonically()][1]
    
    out = []
    prt = { abs(p.pdgid) : p for p in t1.Children }
    b = prt[5]
    lep = [prt[i] for i in [11, 13, 15] if i in prt][0]
    nu = [prt[i] for i in [12, 14, 16] if i in prt][0]
    out.append([b, lep, nu, t1])
    
    prt = { abs(p.pdgid) : p for p in t2.Children }
    b = prt[5]
    lep = [prt[i] for i in [11, 13, 15] if i in prt][0]
    nu = [prt[i] for i in [12, 14, 16] if i in prt][0]
    out.append([b, lep, nu, t2])

    return out

#direc = "/home/tnom6927/Downloads/samples/tttt/QU_0.root"
#Ana = Analysis()
#Ana.InputSample("bsm700", direc)
#Ana.Event = Event
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.Launch()
#
#
#vl = {"b" : [], "lep" : [], "nu" : [], "ev" : [], "t" : []}
#for i in Ana:
#    ev = i.Trees["nominal"]
#    tops = [ t for t in ev.Tops if t.DecayLeptonically()]
#
#    if len(tops) == 2:
#        k = ParticleCollectors(ev)
#        vl["b"].append(  [k[0][0], k[1][0]])
#        vl["lep"].append([k[0][1], k[1][1]])
#        vl["nu"].append( [k[0][2], k[1][2]])
#        vl["t"].append(  [k[0][3], k[1][3]])
#        vl["ev"].append(ev)

#PickleObject(vl, "TMP")
vl = UnpickleObject("TMP")
s_ = DoubleNeutrinoPyT(vl["b"], vl["lep"], vl["ev"])
s_r = DoubleNeutrino(vl["b"][0], vl["lep"][0], vl["ev"][0])

print("")
print(s_r.S[0])
print("")
print(s_r.S[1])

print("----")
print(s_[0][0])
print("")
print(s_[1][0])

print(s_[2][0])



##k = 0
#for i, j in zip(s_[0], s_[1]):
#    print("---")
#    i[2, :] = 0
#    i[2, 2] = 1
#    print(i)
#    torch.linalg.inv(i)
#
#    j[2, :] = 0
#    j[2, 2] = 1
#    print(j)
#    torch.linalg.inv(j)
#    print(k) 
#
#    k+=1
