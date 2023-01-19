import ROOT as r
import torch
import NuR.Physics.Floats as F
import NuR.Physics.Tensors as T
import NuR.Sols.Floats as FS
import NuR.Sols.Tensors as TS
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
    x0_PyT = FS.x0Polar(mu_or_b.pt, mu_or_b.eta, mu_or_b.phi, mu_or_b.e, mH, mL, "cuda").tolist()[0][0]
    CompareNumerical(x0_root, x0_PyT, "x0")

def TestBeta(p):
    part_PyT = F.BetaPolar(p.pt, p.eta, p.phi, p.e, "cuda").tolist()[0][0]

    part_root = r.TLorentzVector()
    part_root.SetPtEtaPhiE(p.pt, p.eta, p.phi, p.e)
    part_root = part_root.Beta()

    CompareNumerical(part_root, part_PyT, "Beta")

def TestSValues(b, muon, mTop, mW, mNu):
    import time 
    from statistics import mean
    
    col = []
    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    muon_root = r.TLorentzVector()
    muon_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)
    
    its = 10000
    t1 = time.time()
    for i in range(its):
        c = r.Math.VectorUtil.CosTheta(b_root, muon_root)
        s = math.sqrt(1 - c**2)
        Bb, Bm = b_root.Beta(), muon_root.Beta() 
   
        x0p = - (mTop**2 - mW**2 - b_root.M2())/(2*b_root.E())
        x0 = - (mW**2 - muon_root.M2() - mNu**2)/(2* muon_root.E())

        Sx_root = (x0 * Bm - muon_root.P()*(1 - Bm **2)) / Bm **2
        Sy_root = (x0p / Bb - c * Sx_root) / s

    t2 = time.time()
    print("Time: (" + str(its) + ") ", t2 - t1)

    _b = T.ToPxPyPzE(torch.tensor([[b.pt, b.eta, b.phi, b.e] for i in range(its)]))
    _mu = T.ToPxPyPzE(torch.tensor([[muon.pt, muon.eta, muon.phi, muon.e] for i in range(its)]))

    _mW = torch.tensor([[mW] for i in range(its)])
    _mTop = torch.tensor([[mTop] for i in range(its)])
    _mNu = torch.tensor([[mNu] for i in range(its)])
    
    t1 = time.time()
    x = TS.SxSyCartesian(_b, _mu, _mTop, _mW, _mNu)
    t2 = time.time()
    print("Time: (" + str(its) +") ", t2 - t1)

    t1 = time.time()
    Sx_PyT = FS.SxPolar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, mW, mNu, "cpu")
    Sy_PyT = FS.SyPolar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, mTop, mW, mNu, "cpu")
    CompareNumerical(Sx_root, Sx_PyT.tolist()[0][0], "Sx")
    CompareNumerical(Sy_root, Sy_PyT.tolist()[0][0], "Sy")

def TestEps_W_Omega(b, muon, mW, mN):
    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    muon_root = r.TLorentzVector()
    muon_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)
    
    CosTheta = r.Math.VectorUtil.CosTheta(b_root, muon_root)
    SinTheta = math.sqrt(1 - CosTheta**2)

    eps2_root = ( mW**2 - mN**2 )*( 1 - muon_root.Beta()**2 )
    eps2_PyT = FS.Eps2Polar(muon.pt, muon.eta, muon.phi, muon.e, mW, mN, "cpu").tolist()[0][0]
    CompareNumerical(eps2_root, eps2_PyT, "eps2")

    w = ( ( muon_root.Beta()/b_root.Beta() ) - CosTheta ) / SinTheta
    w_PyT = FS.wPolar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, 1, "cpu").tolist()[0][0]
    CompareNumerical(w, w_PyT, "w")

    w_ = ( - ( muon_root.Beta()/b_root.Beta() ) - CosTheta ) / SinTheta
    w__PyT = FS.wPolar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, -1, "cpu").tolist()[0][0]
    CompareNumerical(w_, w__PyT, "w_")

    Om2 = w**2 + 1 - muon_root.Beta()**2 
    Om2_PyT = FS.Omega2Polar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, "cpu").tolist()[0][0]
    CompareNumerical(Om2, Om2_PyT, "Omega2")






