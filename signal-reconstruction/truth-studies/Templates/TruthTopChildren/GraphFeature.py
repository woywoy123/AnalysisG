def nNeutrinos(ev):
    x = [1 for i in ev.TopChildren if abs(i.pdgid) in [12, 14, 16]]
    return sum(x)

def nLeptons(ev):
    x = [1 for i in ev.TopChildren if abs(i.pdgid) in [11, 13, 15]]
    return sum(x)

def MET(ev):
    return float(ev.met)

def MET_Phi(ev):
    return float(ev.met_phi)

def Signal(ev):
    return 1 if sum([1 for i in ev.Tops if i.FromRes]) > 0 else 0

def nTops(ev):
    return len(ev.Tops)


