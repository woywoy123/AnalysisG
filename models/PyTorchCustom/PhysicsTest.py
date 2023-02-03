import torch
import ROOT as r
from Checks import *
from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
import PyC.Physics.Tensors.Cartesian as TC
import PyC.Physics.Tensors.Polar as TP

def ParticleToROOT(part):
    _part = r.TLorentzVector()
    _part.SetPtEtaPhiE(part.pt, part.eta, part.phi, part.e)
    return _part

def ParticleToTorch(part):
    tx = torch.tensor([[part._px]], device = "cuda"); 
    ty = torch.tensor([[part._py]], device = "cuda"); 
    tz = torch.tensor([[part._pz]], device = "cuda"); 
    te = torch.tensor([[part._e]], device = "cuda"); 
    return tx, ty, tz, te



Ana = Analysis()
Ana.InputSample("bsm4t-1000", "/home/tnom6927/Downloads/samples/tttt/DAOD_TOPQ1.21955717._000003.root")
Ana.Event = Event 
Ana.EventStop = 100
Ana.EventCache = True 
Ana.DumpPickle = True 
Ana.Launch()

for i in Ana:
    event = i.Trees["nominal"]
    top = event.Tops[0]
    tx, ty, tz, te = ParticleToTorch(top)
    top_r = ParticleToROOT(top)
    
    theta_r = top_r.Theta()
    theta_p = TC.Theta(tx, ty, tz)
    AssertEquivalenceRecursive([[theta_r]], theta_p) 
    
    p_r = top_r.P()
    p_p = TC.P(tx, ty, tz)
    AssertEquivalenceRecursive([[p_r]], p_p) 

    beta_r = top_r.Beta()
    beta_p = TC.Beta(tx, ty, tz, te)
    AssertEquivalenceRecursive([[beta_r]], beta_p) 

    m_r = top_r.M()
    m_p = TC.M(tx, ty, tz, te)
    AssertEquivalenceRecursive([[m_r]], m_p) 

    mt_r = top_r.Mt()
    mt_p = TC.Mt(tz, te)
    AssertEquivalenceRecursive([[mt_r]], mt_p)   
