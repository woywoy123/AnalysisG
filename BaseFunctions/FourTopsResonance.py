from BaseFunctions.IO import *
from BaseFunctions.Physics import *
import numpy as np

def ResonanceFromTruthTops(file_dir):

    tree = "nominal"
    branches = ["truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e", "top_FromRes"]
    fi = FastReading(file_dir)
    fi.ReadBranchFromTree(tree, branches)
    fi.ConvertBranchesToArray(core = 12)
    
    truth_top = fi.ArrayBranches
    
    SpecMass = []
    SignMass = []
    TopMass = []
    for i in truth_top:
        for x in range(len(truth_top[i][branches[0]])):
            pt = truth_top[i][branches[0]][x]
            eta = truth_top[i][branches[1]][x]
            phi = truth_top[i][branches[2]][x]
            e = truth_top[i][branches[3]][x]
            bv = BulkParticleVector(pt, eta, phi, e) 
            Res = truth_top[i][branches[4]][x]
            
            masked = np.ma.masked_array(bv, mask = Res) 
            signals = masked[masked.mask == True]     
            spectator = masked[masked.mask == False] 
            signals.mask = np.ma.nomask

            VecRes = SumVectors(signals)
            VecSpec = SumVectors(spectator)
            
            SpecMass.append(VecSpec.mass() / 1000.)
            SignMass.append(VecRes.mass() / 1000.)
            
            for p in bv:
                TopMass.append(p.mass() / 1000.)
    
    return TopMass, SignMass, SpecMass


def SignalTopsFromChildren(file_dir):
    
    tree = "nominal"
    mask = ["top_FromRes"]
    child = ["truth_top_child_pdgid", "truth_top_child_eta", "truth_top_child_phi", "truth_top_child_pt", "truth_top_child_e"]
    
    child_initState = ["top_initialState_child_pdgid", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt", "truth_top_initialState_child_e"]

    
    SignalMass = []
    SignalDaughterPDGs = []
    SignalDaughterMass = []
    TopMass = []
  
    res = SignalSpectator(mask, tree, child, file_dir)
    for i in res.EventContainer:
        Z_ = Particle()
        dic = i["FakeParents"]
        for t in dic:
            TopMass.append(dic[t].Mass)
                
            if dic[t].IsSignal == 1:
                Z_.AddProduct(dic[t])
                
                for h in dic[t].DecayProducts:
                    SignalDaughterPDGs.append(h.PDGID)
                    SignalDaughterMass.append(h.Mass * 1000)
        
        Z_.ReconstructFourVectorFromProducts()
        SignalMass.append(Z_.Mass)
    
        
    return SignalMass, SignalDaughterMass, SignalDaughterPDGs, TopMass




