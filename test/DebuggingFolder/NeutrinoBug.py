from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Particles.Particles import Neutrino
from AnalysisTopGNN.Events import Event

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


direc = "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/"
Ana = Analysis()
Ana.InputSample("bsm1000", direc)
Ana.Event = Event
Ana.EventCache = True
Ana.DumpPickle = True 
Ana.chnk = 100
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
        
        _ti = 0
        nu1 = Neutrino()
        nu1.px = k[_ti][2]._px
        nu1.py = k[_ti][2]._py
        nu1.pz = k[_ti][2]._pz
        nu1.e = k[_ti][2]._e
        
        print([l.pdgid for l in k[_ti][3].Children])
        reco_top = sum([k[_ti][0], k[_ti][1], k[_ti][2]])
        tru_top = k[_ti][3]
        sum_top = sum([nu1, k[_ti][0], k[_ti][1]])
        print(reco_top.Mass)
        print(tru_top.Mass) 
        print(sum_top.Mass)
