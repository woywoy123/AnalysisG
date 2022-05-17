

def MissingET(ev):
    return ev.met

def MissingPhi(ev):
    return ev.met_phi

def Mu(ev):
    return ev.mu

def MuActual(ev):
    return ev.mu_actual

def NJets(ev):
    return len(ev.Jets)

def Signal(ev):
    if len(ev.TruthTops) == 4:
        return float(1)
    return float(0.)
