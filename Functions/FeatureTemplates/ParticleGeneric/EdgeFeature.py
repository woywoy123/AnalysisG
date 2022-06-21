import math
import LorentzVector as LV

# ================== Kinematics ================ #
def delta_pT(a, b):
    return float(a.pt - b.pt)

def delta_eta(a, b):
    return float(a.eta - b.eta)

def delta_phi(a, b):
    return float(a.phi - b.phi)

def delta_energy(a, b):
    return float(a.energy - b.energy)

def delta_R(a, b):
    return float(a.DeltaR(b))

def mass(a, b):
    m = a.CalculateMass([a, b])
    return float(m.Mass_GeV)


# ================== Truth ================ #
def Index(a, b):
    if a.Index == b.Index:
        return float(1)
    return float(0)

def Expected_pT(a, b):
    if hasattr(a, "exp_pT") == False:
        v1 = LV.ToPxPyPzE(a.pt, a.eta, a.phi, a.e, "cpu")
        a.tmp = LV.TensorToPtEtaPhiE(v1)
    if a == b or a.Index != b.Index:
        return 
    Pmua = LV.TensorToPxPyPzE(a.tmp) 
    Pmua += LV.ToPxPyPzE(b.pt, b.eta, b.phi, b.e, "cpu")
    a.tmp = LV.TensorToPtEtaPhiE(Pmua)
    a.exp_pT = a.tmp[0]
    a.exp_pT = float(a.exp_pT[0])
