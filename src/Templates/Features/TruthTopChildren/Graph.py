def MET(ev):
    return float(ev.met)

def MET_Phi(ev):
    return float(ev.met_phi)

def Signal(ev):
    return 1 if sum([1 for i in ev.Tops if i.FromRes]) > 0 else 0


