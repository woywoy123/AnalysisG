import ROOT as r
from LorentzVector import ToPx, ToPy, ToPz
import math 

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


