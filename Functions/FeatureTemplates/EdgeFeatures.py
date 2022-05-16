import torch
import LorentzVector

def d_r(a, b):
    return float(a.DeltaR(b))

def mass(a, b):
    t_i = LorentzVector.ToPxPyPzE(a.pt, a.eta, a.phi, a.e, "cpu")
    t_j = LorentzVector.ToPxPyPzE(b.pt, b.eta, b.phi, b.e, "cpu")
    return float(LorentzVector.GetMass(t_i + t_j))

def dphi(a, b):
    return float(abs(a.phi - b.phi))

def Signal(a, b):
    return float(a.FromRes * b.FromRes)



