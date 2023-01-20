import ROOT as r
import torch
import NuR.Physics.Floats as F
import NuR.Physics.Tensors as T
#import NuR.Sols.Floats as FS
#import NuR.Sols.Tensors as TS
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

def TestIntermediateValues(b, muon, mT, mW, mN):
    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    mu_root = r.TLorentzVector()
    mu_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)

    CompareNumerical(b_root.Px(), F.ToPx(b.pt, b.phi, "cpu").tolist()[0][0], "Px")
    CompareNumerical(b_root.Py(), F.ToPy(b.pt, b.phi, "cpu").tolist()[0][0], "Py")
    CompareNumerical(b_root.Pz(), F.ToPz(b.pt, b.eta, "cpu").tolist()[0][0], "Pz")
    
    CompareNumerical(b_root.P(), F.PPolar(b.pt, b.eta, b.phi, "cpu").tolist()[0][0], "P")

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

def TestxyZ2(b, muon, mT, mW, mN):
    from time import time 
    its = 100000
    t1 = time() 
    b_root = r.TLorentzVector()
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    mu_root = r.TLorentzVector()
    mu_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)
    for i in range(its): 
        c_root = r.Math.VectorUtil.CosTheta(b_root, mu_root)
        s_root = math.sqrt(1 - c_root**2)
   
        x0p = - (mT**2 - mW**2 - b_root.M2())/(2*b_root.E())
        x0 = - (mW**2 - mu_root.M2() - mN**2)/(2* mu_root.E())
        Bb , Bm = b_root.Beta(), mu_root.Beta()

        Sx_root = (x0 * Bm - mu_root.P()*(1 - Bm **2)) / Bm **2
        Sy_root = (x0p / Bb - c_root * Sx_root) / s_root

        eps2 = (mW**2 - mN**2) * (1 - Bm**2)

        w = (Bm / Bb - c_root) / s_root
        w_ = (-Bm / Bb - c_root) / s_root

        Om2 = w**2 + 1 - Bm **2

        x1 = Sx_root - (Sx_root + w * Sy_root)/Om2
        y1 = Sy_root - (Sx_root + w * Sy_root) * w/Om2
        
        Z = x1**2 * Om2 - (Sy_root - w * Sx_root)**2 - (mW**2 - x0**2 - eps2)
    t2 = time() 
    print("Original time: (" + str(its) + ")", t2-t1)

    _b  = torch.tensor([[b.pt, b.eta, b.phi, b.e] for i in range(its)], device = "cpu")
    _mu = torch.tensor([[muon.pt, muon.eta, muon.phi, muon.e] for i in range(its)], device = "cpu")
    _mT = torch.tensor([[mT] for i in range(its)], device = "cpu")
    _mW = torch.tensor([[mW] for i in range(its)], device = "cpu")
    _mN = torch.tensor([[mN] for i in range(its)], device = "cpu")
    
    t1 = time()
    t = TS.AnalyticalSolutionsPolar(_b, _mu, _mT, _mW, _mN)
    t2 = time()
    print("C++ time: (" + str(its) + ")", t2-t1)
    print(t)

    k = [c_root, s_root, x0, x0p, Sx_root, Sy_root, w, w_, x1, y1, Z, Om2, eps2]
    keys = ["cos", "sin", "x0", "x0p", "Sx", "Sy", "w", "w-", "x", "y", "Z2", "Omega2", "eps2"]

    d = FS.AnalyticalSolutionsPolar(b.pt, b.eta, b.phi, b.e, muon.pt, muon.eta, muon.phi, muon.e, mT, mW, mN, "cpu").tolist()[0]; 
    d = { keys[i] : [ d[i], k[i] ] for i in range(len(keys)) }
    
    for i in d:
        CompareNumerical(d[i][1], d[i][0], i)

def TestS2V0(sxx, sxy, syx, syy, met, phi):
    import numpy as np
    import NuR.SingleNu.Floats as Sf
    import torch.nn.functional as G
    
    sigma2 = [[sxx, sxy], [syx, syy]]
    sigma2 = np.linalg.inv(sigma2)
    sigma2 = np.vstack([sigma2, [0, 0]])
    sigma2 = sigma2.T
    sigma2 = np.vstack([sigma2, [0, 0, 0]])
    print(sigma2)
    
    print("--------")
    n = 5
    matrix = Sf.Sigma2_F(sxx, sxy, syx, syy, "cuda")
    sxx_ = torch.tensor([sxx for i in range(n) ], dtype = np.float, device = "cuda").view(-1, 1)
    syx_ = torch.tensor([syx for i in range(n) ], dtype = np.float, device = "cuda").view(-1, 1)
    sxy_ = torch.tensor([sxy for i in range(n) ], dtype = np.float, device = "cuda").view(-1, 1)
    syy_ = torch.tensor([syy for i in range(n) ], dtype = np.float, device = "cuda").view(-1, 1)
    matrix_ = Sf.Sigma2_T(sxx_, sxy_, syx_, syy_)
    print(matrix_)
    print("--------") 

    metx = F.ToPx(met, phi, "cuda")
    metx = torch.cat([metx for i in range(n)], dim = 0)
    mety = F.ToPy(met, phi, "cuda")
    mety = torch.cat([mety for i in range(n)], dim = 0)   
    _tmp = torch.cat([torch.tensor([[0.]], device = "cuda") for i in range(n)], dim = 0)
    _tmp2 = torch.cat([torch.tensor([[1.]], device = "cuda") for i in range(n)], dim = 0)
    _tmp2 = torch.cat([_tmp, _tmp, _tmp2], dim = -1).view(-1, 1, 3)
    _tmp = torch.cat([metx, mety, _tmp], dim = -1)

    V0 = np.outer([metx.tolist()[0][0] , mety.tolist()[0][0] , 0], [0, 0, 1])
    print(V0)
    
    #print(torch.einsum('ij,ijk->ijk', _tmp, _tmp2))

    V0_ = Sf.V0_T(metx, mety)
    print(V0_)
