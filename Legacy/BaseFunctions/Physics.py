import numpy as np
from skhep.math.vectors import LorentzVector

def ParticleVector(pt, eta, phi, en):

    v = LorentzVector()
    v.setptetaphie(pt, eta, phi, en)
    return v; 

def BulkParticleVector(pt, eta, phi, en):
    Output = []
    for i in range(len(pt)):
        v = ParticleVector(pt[i], eta[i], phi[i], en[i])
        Output.append(v)
        
    return Output

def SumVectors(vector):
    v = LorentzVector()
    for i in vector:
        v += i
    return v



