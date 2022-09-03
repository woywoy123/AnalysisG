def delta_pT(a, b):
    return float(a.PT - b.PT)

def delta_eta(a, b):
    return float(a.Eta - b.Eta)

def delta_phi(a, b):
    return float(a.Phi - b.Phi)

def delta_energy(a, b):
    return float(a.E - b.E)
def delta_Index(a,b):
    if a.Index == b.Index:
        return 1
    else:
        return 0
