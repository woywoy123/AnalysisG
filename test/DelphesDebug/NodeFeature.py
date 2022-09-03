def pdgid(a):
    return float(a.PID)
def PT(a):
    return 1 #float(a.PT)
def Eta(a): 
    return float(a.Eta)
def Phi(a):
    return float(a.Phi)
def Energy(a):
    return float(a.E)
def Mass(a):
    return float(a.Mass)
## Truth ##
def Index(a):
    if a.Index < 0:
        return 1
    else:
        return 0
