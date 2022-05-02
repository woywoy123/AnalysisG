

def MissingET(ev):
    return ev.met

def MissingPhi(ev):
    return ev.met_phi

def MU(ev):
    return ev.mu

def NJets(ev):
    return len(ev.Jets)

def Resonance(ev):
    if len(ev.TruthTops) == 4:
        return float(1)
    return float(0.)
