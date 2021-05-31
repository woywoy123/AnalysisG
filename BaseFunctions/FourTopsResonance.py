from BaseFunctions.IO import *
from BaseFunctions.Physics import *
import numpy as np
import pickle

def ResonanceFromTruthTops(file_dir):

    tree = "nominal"
    branches = ["truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e", "top_FromRes"]
    fi = FastReading(file_dir)
    fi.ReadBranchFromTree(tree, branches)
    fi.ConvertBranchesToArray()
    
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

    res = SignalSpectator(mask, tree, child, file_dir)
    res_init = SignalSpectator(mask, tree, child_initState, file_dir)

    #PickleObject(res_s, "top_child")
    #PickleObject(res_init_s, "top_child_initState")
    #
    #res = UnpickleObject("top_child")
    #res_init = UnpickleObject("top_child_initState")

    SignalMass = []
    SignalDaughterPDGs = []
    SignalDaughterMass = []
    TopMass = []
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
 
    init_SignalMass = []
    init_SignalDaughterPDGs = []
    init_SignalDaughterMass = []
    init_TopMass = [] 
    for i in res_init.EventContainer:
        Z_ = Particle()
        dic = i["FakeParents"]
        for t in dic:
            init_TopMass.append(dic[t].Mass)
                
            if dic[t].IsSignal == 1:
                Z_.AddProduct(dic[t])
                
                for h in dic[t].DecayProducts:
                    init_SignalDaughterPDGs.append(h.PDGID)
                    init_SignalDaughterMass.append(h.Mass * 1000)
        
        Z_.ReconstructFourVectorFromProducts()
        init_SignalMass.append(Z_.Mass)
    
    Output = {}
    Output["SGMass"] = SignalMass
    Output["SGMass_init"] = SignalMass
    Output["SGDaughterM"] = SignalDaughterMass
    Output["SGDaughterM_init"] = init_SignalDaughterMass
    Output["SGDaughterPDG"] = SignalDaughterPDGs
    Output["SGDaughterPDG_init"] = init_SignalDaughterPDGs
    Output["TopMass"] = TopMass
    Output["TopMass_init"] = init_TopMass   

    return Output




