def TruthNodeFromRes(a):
    return a.FromResonance

def TruthResonancePair(particle1, particle2):
    if particle1.FromResonance == particle2.FromResonance:
        return 1
    else:
        return 0

def SignalEvent(ev):
    sig = [t.FromResonance for t in ev.Tops]
    if len(sig) > 0:
        return 1
    return 0
        
def nJets(ev):
    return ev.nJets 

def Energy(particle):
    return particle.e

def Phi(particle):
    return particle.phi

def Eta(particle):
    return particle.eta

def PT(particle):
    return particle.pt

def EdgeMass(a1, a2):
    p = a1 + a2
    return p.Mass
