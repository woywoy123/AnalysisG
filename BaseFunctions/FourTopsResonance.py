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
    def FillMaps(Container, Sig):
        SignalMass = []
        SignalDaughterPDGs = []
        SignalDaughterMass = []
        TopMass = []
        
        DaughterMassPDG = {}
        for i in Container.EventContainer:
    
            FP = i["FakeParents"]
            SIG = i["Signal"]
            SPEC = i["Spectator"]
             
            Z_ = Particle()
            for j in FP:
    
                # The four tops are going to be tops and should have the associated mass 
                if FP[j].IsSignal != Sig:
                    continue
                Z_.AddProduct(FP[j])
                TopMass.append(FP[j].Mass)
                
                # Record the children from the Signal Daughters
                Child = FP[j].DecayProducts
                for c in Child:
                    try:
                        DaughterMassPDG[c.PDGID].append(c.Mass*1000) 
                    except KeyError:
                        DaughterMassPDG[c.PDGID] = []
                        DaughterMassPDG[c.PDGID].append(c.Mass*1000)
                    
                    SignalDaughterMass.append(c.Mass*1000)
                    SignalDaughterPDGs.append(c.PDGID)
                
            Z_.ReconstructFourVectorFromProducts()
            SignalMass.append(Z_.Mass)
        return SignalMass, SignalDaughterPDGs, SignalDaughterMass, TopMass, DaughterMassPDG


    tree = "nominal"
    mask = ["top_FromRes"]
    child = ["truth_top_child_pdgid", "truth_top_child_eta", "truth_top_child_phi", "truth_top_child_pt", "truth_top_child_e"]
    child_initState = ["top_initialState_child_pdgid", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt", "truth_top_initialState_child_e"]

    res = SignalSpectator(mask, tree, child, file_dir)
    res_init = SignalSpectator(mask, tree, child_initState, file_dir)

    PickleObject(res, "top_child")
    PickleObject(res_init, "top_child_initState")
    
    res = UnpickleObject("top_child")
    res_init = UnpickleObject("top_child_initState")

    SignalMass, SignalDaughterPDGs, SignalDaughterMass, TopMass, DaughterMassPDG = FillMaps(res, 1)
    init_SignalMass, init_SignalDaughterPDGs, init_SignalDaughterMass, init_TopMass, init_DaughterMassPDG = FillMaps(res_init, 1)

    SpecMass, SpecDaughterPDGs, SpecDaughterMass, SpecTopMass, SpecDaughterMassPDG = FillMaps(res, 0)
    init_SpecMass, init_SpecDaughterPDGs, init_SpecDaughterMass, init_SpecTopMass, init_SpecDaughterMassPDG = FillMaps(res_init, 0)

    Output = {}
    Output["SGMass"] = SignalMass
    Output["SGMass_init"] = init_SignalMass
    Output["SGDaughterM"] = SignalDaughterMass
    Output["SGDaughterM_init"] = init_SignalDaughterMass
    Output["SGDaughterPDG"] = SignalDaughterPDGs
    Output["SGDaughterPDG_init"] = init_SignalDaughterPDGs
    Output["TopMass"] = TopMass
    Output["TopMass_init"] = init_TopMass   
    Output["SGDMassPDG"] = DaughterMassPDG
    Output["SGDMassPDG_init"] = init_DaughterMassPDG

    Output["SpecMass"] = SpecMass
    Output["SpecMass_init"] = init_SpecMass
    Output["SpecDaughterM"] = SpecDaughterMass
    Output["SpecDaughterM_init"] = init_SpecDaughterMass
    Output["SpecDaughterPDG"] = SpecDaughterPDGs
    Output["SpecDaughterPDG_init"] = init_SpecDaughterPDGs
    Output["SpecTopMass"] = SpecTopMass
    Output["SpecTopMass_init"] = init_SpecTopMass   
    Output["SpecMassPDG"] = SpecDaughterMassPDG
    Output["SpecMassPDG_init"] = init_SpecDaughterMassPDG
    return Output

def ChildToTruthJet(file_dir):
    
    tree = "nominal"
    mask = ["top_FromRes"]
    child_initState = ["top_initialState_child_pdgid", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt", "truth_top_initialState_child_e"]
    truth_jets = ["truthjet_flavour", "truthjet_e", "truthjet_phi", "truthjet_eta", "truthjet_pt", "met_met"] 
    
    #jet_tF = EventJetCompiler(tree, truth_jets, file_dir)




