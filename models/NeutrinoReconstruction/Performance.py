from neutrino_momentum_reconstruction_python3 import singleNeutrinoSolution
from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event 
import ROOT as r
import torch
import NuR.SingleNu.Floats as Sf
import NuR.Physics.Floats as F

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

def _MakeTensor(val, number, device = "cpu", dtp = torch.double):
    return torch.tensor([val for i in range(number)], device = device, dtype = dtp)

direc = "/home/tnom6927/Downloads/samples/SingleTop/m700/DAOD_TOPQ1.21955710._000003.root"
Ana = Analysis()
Ana.InputSample("bsm700", direc)
Ana.Event = Event
Ana.EventCache = True
Ana.DumpPickle = True 
Ana.Launch()

for i in Ana:
    ev = i.Trees["nominal"]
    tops = [ t for t in ev.Tops if t.DecayLeptonically()]
    if len(tops) == 1:
        break

singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]
singlelepton = {abs(i.pdgid) : i for i in singlelepton}
b = singlelepton[5]
muon = singlelepton[13]
nu = singlelepton[14]

r_b = r.TLorentzVector()
r_b.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

r_mu = r.TLorentzVector()
r_mu.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)

mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass
Sxx = 100
Sxy = 0
Syx = 0
Syy = 100

x = F.ToPx(ev.met, ev.met_phi, "cpu").tolist()[0][0]
y = F.ToPy(ev.met, ev.met_phi, "cpu").tolist()[0][0]
sol = singleNeutrinoSolution(r_b, r_mu, x, y, [[Sxx, Sxy], [Syx, Syy]])

device = "cuda"
n = 1
_b  = _MakeTensor([b.pt, b.eta, b.phi, b.e], n, device)
_mu = _MakeTensor([muon.pt, muon.eta, muon.phi, muon.e], n, device)
_mT = _MakeTensor([mT], n, device)
_mW = _MakeTensor([mW], n, device)
_mN = _MakeTensor([mN], n, device)
_met = _MakeTensor([ev.met], n, device)
_phi = _MakeTensor([ev.met_phi], n, device)
_Sxx = _MakeTensor([Sxx ], n, device)
_Sxy = _MakeTensor([Sxy ], n, device) 
_Syx = _MakeTensor([Syx ], n, device) 
_Syy = _MakeTensor([Syy ], n, device)

_sol = Sf.SolT(_b, _mu, _mT, _mW, _mN, _met, _phi, _Sxx, _Sxy, _Syx, _Syy)
print(_sol)
print(sol.nu, sol.chi2)

print("---")
x = F.ToPxPyPzE(nu.pt, nu.eta, nu.phi, nu.e, "cpu")
Px = float(x[0][0])
Py = float(x[0][1])
Pz = float(x[0][2])
print(Px, Py, Pz)

sol = singleNeutrinoSolution(r_b, r_mu, Px, Py, [[Sxx, Sxy], [Syx, Syy]])
print(sol.nu)
