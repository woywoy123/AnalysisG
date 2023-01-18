import ROOT as r
import torch
import Floats as F
import Tensors as T
import math

def CompareNumerical(r_ori, r_pyt, string):
    print("(" + string + ") -> Original: ", r_ori, " ||  Pytorch: ", r_pyt, " || Error (%): ", 100*abs(r_pyt - r_ori)/r_ori)

def TestFourVector(part):
    p_ = r.TLorentzVector()
    p_.SetPtEtaPhiE(part.pt, part.eta, part.phi, part.e)
    
    p_PyT_f = F.ToPxPyPzE(part.pt, part.eta, part.phi, part.e, "cuda").tolist()[0]
    CompareNumerical(p_.Px(), p_PyT_f[0], "Px")
    CompareNumerical(p_.Py(), p_PyT_f[1], "Py")
    CompareNumerical(p_.Pz(), p_PyT_f[2], "Pz")
    CompareNumerical(p_.E(), p_PyT_f[3], "E")

def TestCosTheta(b, muon):
    b_PyT = F.ToPxPyPzE(b.pt, b.eta, b.phi, b.e, "cuda")
    muon_PyT = F.ToPxPyPzE(muon.pt, muon.eta, muon.phi, muon.e, "cuda")

    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    muon_root = r.TLorentzVector()
    muon_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)

    cos_root = r.Math.VectorUtil.CosTheta(b_root, muon_root)
    cos_PyT = T.CosThetaCartesian(b_PyT, muon_PyT).tolist()[0][0]
    CompareNumerical(cos_root, cos_PyT, "CosTheta")

    sin_root = math.sqrt(1 - cos_root**2)
    sin_PyT = T.SinThetaCartesian(b_PyT, muon_PyT).tolist()[0][0]
    CompareNumerical(sin_root, sin_PyT, "SinTheta")

def Testx0(mH, mL, mu_or_b):
    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(mu_or_b.pt, mu_or_b.eta, mu_or_b.phi, mu_or_b.e)
    
    x0_root = - (mH**2 - mL**2 - b_root.M2())/(2*b_root.E())
    x0_PyT = F.x0(mu_or_b.pt, mu_or_b.eta, mu_or_b.phi, mu_or_b.e, mH, mL, "cuda")
    print(x0_root, x0_PyT)
