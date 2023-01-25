from neutrino_momentum_reconstruction_python3 import singleNeutrinoSolution
from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event 
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
import ROOT as r
import torch
import NuR.SingleNu.Floats as Sf
import NuR.Physics.Floats as F
from time import time

Sxx = 100
Sxy = 0
Syx = 0
Syy = 100
mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass
device = "cpu"

def CompareNumerical(r_ori, r_pyt, string):
    print("(" + string + ") -> Original: ", r_ori, " ||  Pytorch: ", r_pyt, " || Error (%): ", 100*abs(r_pyt - r_ori)/r_ori)

def CompareListNumerical(r_ori, r_pyt, title = "", string = ""):
    print("-> " + title)
    if string == "":
        for i, j in zip(r_ori, r_pyt):
            delta = float(sum(i - j))
            print("ROOT: ", list(i), "PyTorch: ", list(j), " || Error (%): ", 100*abs(delta/sum(abs(i))))
        print("")
        return 
    for i, j, k in zip(r_ori, r_pyt, string):
        CompareNumerical(i, j, k)
    print("")

def _MakeTensor(val, n, device = "cpu", dtp = torch.double):
    return torch.tensor([val for i in range(n)], device = device, dtype = dtp)


def SingleNeutrino(b, mu, ev):
    r_b = r.TLorentzVector()
    r_b.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)
    
    r_mu = r.TLorentzVector()
    r_mu.SetPtEtaPhiE(mu.pt, mu.eta, mu.phi, mu.e)

    x = F.ToPx(ev.met, ev.met_phi, "cpu").tolist()[0][0]
    y = F.ToPy(ev.met, ev.met_phi, "cpu").tolist()[0][0]
    return singleNeutrinoSolution(r_b, r_mu, x, y, [[Sxx, Sxy], [Syx, Syy]])
    
def SingleNeutrinoPyT(b, mu, ev):
    _b = torch.tensor([[b_.pt, b_.eta, b_.phi, b_.e] for b_ in b], dtype = torch.double, device = device)
    _mu = torch.tensor([[mu_.pt, mu_.eta, mu_.phi, mu_.e] for mu_ in mu], dtype = torch.double, device = device)

    _met = torch.tensor([[ev_.met] for ev_ in ev], dtype = torch.double, device = device)
    _phi = torch.tensor([[ev_.met_phi] for ev_ in ev], dtype = torch.double, device = device)
    
    n = len(b)
    _mT = _MakeTensor([mT], n, device)
    _mW = _MakeTensor([mW], n, device)
    _mN = _MakeTensor([mN], n, device)
    _Sxx = _MakeTensor([Sxx], n, device)
    _Sxy = _MakeTensor([Sxy], n, device) 
    _Syx = _MakeTensor([Syx], n, device) 
    _Syy = _MakeTensor([Syy], n, device)
    return Sf.SolT(_b, _mu, _mT, _mW, _mN, _met, _phi, _Sxx, _Sxy, _Syx, _Syy)

def ParticleCollectors(ev):
    prt = { abs(p.pdgid) : p for t in ev.Tops for p in t.Children if t.DecayLeptonically() }
    b = prt[5]
    lep = [prt[i] for i in [11, 13, 15] if i in prt][0]
    nu = [prt[i] for i in [12, 14, 16] if i in prt][0]
    t = [ t for t in ev.Tops if t.DecayLeptonically()][0]
    return [b, lep, nu, ev, t]

#direc = "/home/tnom6927/Downloads/samples/SingleTop/m700/DAOD_TOPQ1.21955710._000003.root"
#Ana = Analysis()
#Ana.InputSample("bsm700", direc)
#Ana.Event = Event
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.Launch()
#
#vl = {"b" : [], "lep" : [], "nu" : [], "ev" : [], "t" : []}
#for i in Ana:
#    ev = i.Trees["nominal"]
#    tops = [ t for t in ev.Tops if t.DecayLeptonically()]
#
#    if len(tops) == 1:
#        k = ParticleCollectors(ev)
#        vl["b"].append(k[0])
#        vl["lep"].append(k[1])
#        vl["nu"].append(k[2])
#        vl["ev"].append(k[3])
#        vl["t"].append(k[4])
#
#res = {"Or" : [], "PT" : []}
#t1p = time()    
##for i in range(len(vl["b"])):
##    res["Or"].append(SingleNeutrino(vl["b"][i], vl["lep"][i], vl["ev"][i]))
#t2p = time()
#
#t1c = time()
##for i in range(len(vl["b"])):
#    print(i)

#PickleObject(vl, "TMP")
#vl = UnpickleObject("TMP")
#res["PT"] = SingleNeutrinoPyT([vl["b"][1]]*100, [vl["lep"][1]]*100, [vl["ev"][1]]*100)
#t2c = time()

#print("Python: ", t2p - t1p)
#print("C++: ", t2c - t1c)


### Debugging 
vl = UnpickleObject("TMP")
#res["PT"] = SingleNeutrinoPyT([vl["b"][1]]*100, [vl["lep"][1]]*100, [vl["ev"][1]]*100)
c = SingleNeutrinoPyT(vl["b"], vl["lep"], vl["ev"])

it = -1
for i in range(43, len(c[0])):
    it += 1

    SingleNeutrino(vl["b"][i], vl["lep"][i], vl["ev"][i]).nu
    print("->", c[0][i])

    print(it)
    print("----")
    break
    
