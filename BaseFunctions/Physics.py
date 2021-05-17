import ROOT

def ParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()
    v.SetCoordinates(pt, eta, phi, en)
    return v; 

def MultiParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()   
    for i in range(len(pt)):
        v = v + ParticleVector(pt[i], eta[i], phi[i], en[i])
    return v; 
