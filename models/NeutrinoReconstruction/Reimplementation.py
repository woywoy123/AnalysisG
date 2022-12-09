import ROOT as r
from LorentzVector import ToPx, ToPy, ToPz
import math 
import numpy as np
import torch

massT = 172.5*1000
massW = 80.385*1000
massNu = 0

def Mag(Vec):
    return math.sqrt(sum([i**2 for i in Vec]))

def MagP(p):
    p_v = [ToPx(p.pt, p.phi), ToPy(p.pt, p.phi), ToPz(p.pt, p.eta), p.e]
    return Mag(p_v)
        

def Dot(Vec1, Vec2):
    return sum([i*j for i, j in zip(Vec1, Vec2)])

def CosTheta(p1, p2):
    p1_v = [ToPx(p1.pt, p1.phi), ToPy(p1.pt, p1.phi), ToPz(p1.pt, p1.eta), p1.e]
    p2_v = [ToPx(p2.pt, p2.phi), ToPy(p2.pt, p2.phi), ToPz(p2.pt, p2.eta), p2.e]
    
    mag_p1 = Mag(p1_v)
    mag_p2 = Mag(p2_v)
    dot = Dot(p1_v, p2_v)
    return float(dot/(mag_p1 * mag_p2))

def Beta(p):
    return math.sqrt(ToPx(p.pt, p.phi)**2 + ToPy(p.pt, p.phi)**2 + ToPz(p.pt, p.eta)**2)/p.e

def Comparison(string, root, pred):
    print(string + " (R) -> ", root)
    print(string + " (P) -> ", pred)
    print("DIFF: ", pred - root)
    print("ERROR (%): ", float((pred-root)/root)*100)
    print("")

def TestNuSolutionSteps(b, muon):
    b_root = r.TLorentzVector() 
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    mu_root = r.TLorentzVector()
    mu_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)
    
    x0p_R = -(massT**2 - massW**2 - b_root.M2())/(2*b_root.E())
    x0p_P = -(massT**2 - massW**2 - (b.CalculateMass()*1000)**2) / (2*b.e)
    Comparison("x0p" , x0p_R, x0p_P)
 
    x0_R = -(massW**2 - mu_root.M2())/(2*mu_root.E())
    x0_P = -(massW**2 - (muon.CalculateMass()*1000)**2) / (2*muon.e)
    Comparison("x0", x0_R, x0_P)

    Beta_b_R = b_root.Beta()
    Beta_b_P = Beta(b)
    Comparison("Beta_b" , Beta_b_R, Beta_b_P)

    Beta_mu_R = mu_root.Beta()
    Beta_mu_P = Beta(muon)
    Comparison("Beta_mu" , Beta_mu_R, Beta_mu_P)

    Sx_R = (x0_R * Beta_mu_R - mu_root.P() * (1 - Beta_mu_R**2)) / (Beta_mu_R**2)
    Sx_P = (x0_P * Beta_mu_P - MagP(muon) * (1 - Beta_mu_P**2)) / (Beta_mu_P**2)
    Comparison("Sx", Sx_R, Sx_P)
   
    costhetaR = r.Math.VectorUtil.CosTheta(b_root, mu_root)
    costhetaP = CosTheta(b, muon)
    Comparison("costheta" , costhetaR, costhetaP)
    Comparison("CosTheta * Sx", costhetaR*Sx_R, costhetaP*Sx_P)
    
    sinthetaR = math.sqrt(1 - costhetaR**2)
    sinthetaP = math.sqrt(1 - costhetaP**2)
    Comparison("SinTheta" , sinthetaR, sinthetaP)

    Sy_R = (float(x0p_R / Beta_mu_R) - costhetaR * Sx_R) / sinthetaR
    Sy_P = (float(x0p_P / Beta_mu_P) - costhetaP * Sx_P) / sinthetaP
    Comparison("Sy", Sy_R, Sy_P)

    eps2_R = (massW**2 - massNu**2) * (1 - Beta_mu_R**2)
    eps2_P = (massW**2 - massNu**2) * (1 - Beta_mu_P**2)
    Comparison("eps2", eps2_R, eps2_P)

    wp_R = (float(Beta_mu_R / Beta_b_R) - costhetaR) / sinthetaR
    wp_P = (float(Beta_mu_P / Beta_b_P) - costhetaP) / sinthetaP
    Comparison("w+", wp_R, wp_P)

    wm_R = (-float(Beta_mu_R / Beta_b_R) - costhetaR) / sinthetaR
    wm_P = (-float(Beta_mu_P / Beta_b_P) - costhetaP) / sinthetaP
    Comparison("w-", wm_R, wm_P)
    
    Omega2_R = wp_R**2 + 1 - Beta_mu_R**2
    Omega2_P = wp_P**2 + 1 - Beta_mu_P**2
    Comparison("Omega2", Omega2_R, Omega2_P)
    
    x1_R = Sx_R - (Sx_R + wp_R * Sy_R)/Omega2_R
    x1_P = Sx_P - (Sx_P + wp_P * Sy_P)/Omega2_P
    Comparison("x1", x1_R, x1_P)

    y1_R = Sy_R - (Sx_R + wp_R * Sy_R)*wp_R/Omega2_R
    y1_P = Sy_P - (Sx_P + wp_P * Sy_P)*wp_P/Omega2_P
    Comparison("y1", y1_R, y1_P)

    Z2_R = x1_R**2 * Omega2_R - (Sy_R - wp_R * Sx_R)**2 - (massW**2 - x0_R**2 - eps2_R)
    Z2_P = x1_P**2 * Omega2_P - (Sy_P - wp_P * Sx_P)**2 - (massW**2 - x0_P**2 - eps2_P)
    Comparison("Z^2", Z2_R, Z2_P)

    Z_R = math.sqrt(max(0, Z2_R))
    Z_P = math.sqrt(max(0, Z2_P))
    Comparison("Z", Z_R, Z_P)







def Rotation_R(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    R_R = c * np.eye(3)
    R_P = c * torch.eye(3)

    #print("")
    #print("------ Rotation ------")
    #print(R_R)
    #print(R_P)
    #print("") 
    for i in [-1, 0, 1]:
        R_R[(axis - i)%3, (axis + i)%3] = i*s + (1 - i*i)
    return R_R


def R_T(b, muon):
    b_root = r.TLorentzVector() 
    b_root.SetPtEtaPhiE(b.pt, b.eta, b.phi, b.e)

    mu_root = r.TLorentzVector()
    mu_root.SetPtEtaPhiE(muon.pt, muon.eta, muon.phi, muon.e)
    
    b_xyz_R = b_root.X(), b_root.Y(), b_root.Z()
    b_xyz_P = ToPx(b.pt, b.phi), ToPy(b.pt, b.phi), ToPz(b.pt, b.eta)
    print("---- Cartesian Vectors (b-quark) ----") 
    print("Cartesian Vector (R): ", b_xyz_R)
    print("Cartesian Vector (P): ", b_xyz_P)
    
    R_z = Rotation(2, -b_root.Phi())
    R_y = Rotation(1, 0.5*math.pi -mu_root.Theta())
    
    x, y, z = R_y.dot(R_z.dot(b_xyz_R))
    print("->", Rotation(0, -math.atan2(z, y)))
    R_x = next(Rotation(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz_R)), ))
    print("+>", R_x)




def TestSingleNeutrinoSolutionSegment(b, muon, METx, METy, Sigma2):
    print("---- INVERSE ----")
    inv_R = np.linalg.inv(Sigma2)
    inv_P = torch.from_numpy(Sigma2).to(dtype = torch.float32).inverse()
    print(inv_R)
    print(inv_P)
    
    print("")
    print("---- STACK 2x3 ----")   
    stack_R = np.vstack([inv_R, [0, 0]]).T
    stack_P = torch.cat([inv_P, torch.tensor([0, 0]).view(-1, 1)], dim = 1)
    print(stack_R)
    print(stack_P)

    print("")
    print("---- STACK 3x3 ----")   
    S2_R = np.vstack([np.vstack([inv_R, [0, 0]]).T, [0, 0, 0]])
    S2_P = torch.cat([stack_P, torch.tensor([0, 0, 0]).view(1, -1)], dim = 0)
    print(S2_R)
    print(S2_P)
    
    print("")
    print("---- V0 ----")
    V0_R = np.outer([ METx, METy, 0 ], [0, 0, 1]) 
    V0_P = torch.outer(torch.tensor([METx, METy, 0]), torch.tensor([0, 0, 1]))
    print(V0_R)
    print(V0_P)
    print("")

    R_T(b, muon)
    



