from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
import PyC.NuSol.CUDA as NuC
import PyC.Transform.Floats as F
from Checks import *
import torch

def ParticleCollectors(ev):
    t1 = [ t for t in ev.Tops if t.LeptonicDecay][0]
    t2 = [ t for t in ev.Tops if t.LeptonicDecay][1]
    
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

def MakeTensor_(inpt, indx):
    return torch.tensor([[ i[indx]._px/1000, i[indx]._py/1000, i[indx]._pz/1000, i[indx]._e/1000 ] for i in inpt], dtype = torch.float64, device = "cuda")



Ana = Analysis()
Ana.InputSample("bsm4t-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
Ana.Event = Event
Ana.EventStop = 100
Ana.EventCache = True
Ana.DumpPickle = True 
Ana.chnk = 100
Ana.VerboseLevel = 2
Ana.Launch()
it = 0
vl = {"b" : [], "lep" : [], "nu" : [], "ev" : [], "t" : []}
for i in Ana:
    ev = i.Trees["nominal"]
    tops = [ t for t in ev.Tops if t.LeptonicDecay]

    if len(tops) == 2:
        k = ParticleCollectors(ev)
        vl["b"].append(  [k[0][0], k[1][0]])
        vl["lep"].append([k[0][1], k[1][1]])
        vl["nu"].append( [k[0][2], k[1][2]])
        vl["t"].append(  [k[0][3], k[1][3]])
        vl["ev"].append(ev)
        it+=1

    
    if it == 1000:
        break

T = SampleTensor(vl["b"], vl["lep"], vl["ev"], vl["t"], "cuda", [[100, 0], [0, 100]])

# Original CUDA version 
sol_tP = NuC.NuPtEtaPhiE(T.b, T.mu, T.met, T.phi, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN, 1e-12)

b1c = MakeTensor_(vl["b"], 0)
mu1c = MakeTensor_(vl["lep"], 0)

metx = torch.tensor([ [F.Px(i.met, i.met_phi)/1000] for i in vl["ev"] ], dtype = torch.float64, device = "cuda")
mety = torch.tensor([ [F.Py(i.met, i.met_phi)/1000] for i in vl["ev"] ], dtype = torch.float64, device = "cuda")

sol_tC = NuC.NuPxPyPzE(b1c, mu1c, metx, mety, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN, 1e-12)
for i, j in zip(sol_tC[2], sol_tP[2]):
    print(i.tolist(), j.tolist())
