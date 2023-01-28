from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event 
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
import ROOT as r
import torch
import NuR.DoubleNu.Floats as Sf
import NuR.Physics.Floats as F
from neutrino_momentum_reconstruction_python3 import doubleNeutrinoSolutions
import numpy as np
from time import time

mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass
device = "cuda"

def CompareNumerical(r_ori, r_pyt, string):
    print("(" + string + ") -> Original: ", r_ori, " ||  Pytorch: ", r_pyt, " || Error (%): ", 100*abs(r_pyt - r_ori)/r_ori)

class SampleTensor:

    def __init__(self, b, mu, ev):
        self.device = device
        self.n = len(b)
        
        self.b = self.MakeKinematics(0, b)
        self.b_ = self.MakeKinematics(1, b)
        self.mu = self.MakeKinematics(0, mu)
        self.mu_ = self.MakeKinematics(1, mu)
        
        self.mT = self.MakeTensor(mT)
        self.mW = self.MakeTensor(mW)
        self.mN = self.MakeTensor(mN)

        self.MakeEvent(ev)

    def MakeKinematics(self, idx, obj):
        return torch.tensor([[i[idx].pt, i[idx].eta, i[idx].phi, i[idx].e] for i in obj], dtype = torch.double, device = self.device)
    
    def MakeEvent(self, obj):
        self.met = torch.tensor([[ev.met] for ev in obj], dtype = torch.double, device = device)
        self.phi = torch.tensor([[ev.met_phi] for ev in obj], dtype = torch.double, device = device)

    def MakeTensor(self, val):
        return torch.tensor([[val] for i in range(self.n)], dtype = torch.double, device = self.device)

class SampleROOT:
    def __init__(self, b, mu, ev):

        self.n = len(ev)
        self.b = [self.MakeKinematics(0, i) for i in b]
        self.b_ = [self.MakeKinematics(1, i) for i in b]       

        self.mu = [self.MakeKinematics(0, i) for i in mu]
        self.mu_ = [self.MakeKinematics(1, i) for i in mu]       

        self.met_x = []
        self.met_y = []
        
        for i in ev:
            x, y = self.MakeEvent(i)
            self.met_x.append(x)
            self.met_y.append(y)

        self.mT = [mT for i in range(self.n)] 
        self.mW = [mT for i in range(self.n)]
        self.mN = [mT for i in range(self.n)]

    def MakeKinematics(self, idx, obj):
        r_ = r.TLorentzVector()
        r_.SetPtEtaPhiE(obj[idx].pt, obj[idx].eta, obj[idx].phi, obj[idx].e)
        return r_

    def MakeEvent(self, obj):
        x = F.ToPx(obj.met, obj.met_phi, "cpu").tolist()[0][0]
        y = F.ToPy(obj.met, obj.met_phi, "cpu").tolist()[0][0]
        return x, y

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

#direc = "/CERN/Samples/Dilepton/ttH_tttt_m1000/"
#Ana = Analysis()
#Ana.InputSample("bsm1000", direc)
#Ana.Event = Event
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.chnk = 1000
#Ana.Launch()
#
#it = 0
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
#
#    if it == 100:
#        break
#    it += 1
#
#PickleObject(vl, "sml")
vl = UnpickleObject("TMP")

T = SampleTensor(vl["b"], vl["lep"], vl["ev"])
R = SampleROOT(vl["b"], vl["lep"], vl["ev"])

print("n-Events: ", T.n)
t1c = time()
s_ = Sf.SolT(T.b, T.b_, T.mu, T.mu_, T.mT, T.mW, T.mN, T.met, T.phi, 1e-12)
t2c = time()
tc = t2c - t1c
print("C++: ", tc, " device: ", device)

s = []
t1p = time()
for i in range(R.n):
    try:
        s.append(doubleNeutrinoSolutions(R.b[i], R.b_[i], R.mu[i], R.mu_[i], R.met_x[i], R.met_y[i]))
    except:
        pass
t2p = time()
tp = t2p - t1p 
print("Python: ", tp)
print("Speed Factor (> 1 is better): ", tp/tc)

s_inv = Sf.SolT(T.b_, T.b, T.mu_, T.mu, T.mT, T.mW, T.mN, T.met, T.phi, 1e-12)
sinv = []
for i in range(R.n):
    try:
        sinv.append(doubleNeutrinoSolutions(R.b_[i], R.b[i], R.mu_[i], R.mu[i], R.met_x[i], R.met_y[i]))
    except:
        pass
