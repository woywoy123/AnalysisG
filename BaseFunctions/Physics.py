import ROOT
import numpy as np

def ParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()
    v.SetCoordinates(pt, eta, phi, en)
    return v; 

def BulkParticleVector(pt, eta, phi, en):
    Output = []
    for i in range(len(pt)):
        v = ParticleVector(pt[i], eta[i], phi[i], en[i])
        Output.append(v)
    
    return np.array(Output)

def SumMultiParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()   
    for i in range(len(pt)):
        v = v + ParticleVector(pt[i], eta[i], phi[i], en[i])
    return v; 

def SumVectors(vector):
    v = ROOT.Math.PtEtaPhiEVector()
    for i in vector:
        v += i
    return v
