def eta(a):
    return float(a.eta)

def pT(a):
    return float(a.pt)

def phi(a):
    return float(a.phi)

def energy(a):
    return float(a.e)

def mass(a):
    return float(a.CalculateMass())

def islepton(a):
    return 1 if a.Type in ["el", "mu"] else 0

def charge(a):
    if "charge" in a.__dict__:
        return a.charge
    return 0

# ===== Truth =====
def mergedTop(a):
    return 1 if len([i for i in a.index if i > -1]) > 1 else 0

def FromTop(a):
    return 1 if len([i for i in a.index if i > -1]) > 1 else 0

def FromRes(a):
    c1 = sum([k.FromRes for i in a.Parent for k in i.Parent])
    return 1 if c1 != 0 else 0


