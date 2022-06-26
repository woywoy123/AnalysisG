

# ================== Kinematics ================ #
def pT(a):
    return float(a.pt)

def eta(a):
    return float(a.eta)

def phi(a):
    return float(a.phi)

def energy(a):
    return float(a.e)

def mass(a):
    a.CalculateMass()
    return float(a.Mass_GeV)

# ================== Truth ================ #
def Index(a):
    return float(a.Index)

def ExpPx(a):
    return float(a.exp_Px)
