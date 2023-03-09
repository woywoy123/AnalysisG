def nJets(ev):
    return len(ev.TruthJets)

def nLeptons(ev):
    return len(ev.Electrons + ev.Muons)

def MET(ev):
    return float(ev.met)

def MET_Phi(ev):
    return float(ev.met_phi)

# ===== Truth =====
def nTops(ev):
    return len(ev.Tops)

def Signal(ev):
    return 1 if len([1 for i in ev.Tops if i.FromRes == 1]) > 0 else 0


