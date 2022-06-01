import torch
import LorentzVector

def eta(a):
    return float(a.eta)

def energy(a):
    return float(a.e)

def pt(a):
    return float(a.pt)

def phi(a):
    return float(a.phi)

def Signal(a):
    return int(a.FromRes)

def Mass(a):
    v = LorentzVector.ToPxPyPzE(a.pt, a.eta, a.phi, a.e, "cpu")
    return float(LorentzVector.MassFromPxPyPzE(v))

def Index(a):
    return float(a.Index + 1)

def Merged(a):
    if a.Type == "truthjet":
        if a.GhostTruthJetMap[0] == -1:
            return 0
        return len(a.GhostTruthJetMap)
    elif a.Type == "jet":
        if a.JetMapGhost[0] == -1:
            return 0
        return len(a.JetMapTops)
    return 0


