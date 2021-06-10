from BaseFunctions.IO import *
from BaseFunctions.Physics import *
from BaseFunctions.EventsManager import *
from BaseFunctions.VariableManager import *
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
    child = ["top_FromRes", "truth_top_child_pdgid", "truth_top_child_eta", "truth_top_child_phi", "truth_top_child_pt", "truth_top_child_e"]
    child_initState = ["top_FromRes", "top_initialState_child_pdgid", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt", "truth_top_initialState_child_e"]

    res = SignalSpectator(tree, child, file_dir)
    res_init = SignalSpectator(tree, child_initState, file_dir)

    #PickleObject(res, "top_child")
    #PickleObject(res_init, "top_child_initState")
    #
    #res = UnpickleObject("top_child")
    #res_init = UnpickleObject("top_child_initState")

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
    
    Event_Branches = ["eventNumber", "met_met", "met_phi"] 
    Map = BranchVariable(file_dir, tree, Event_Branches).EventObjectMap

    truth_tops = ["truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e", "truth_top_charge"] 
    truth_4t = TruthCompiler(file_dir, tree, truth_tops)
    truth_4t.GenerateEvents()

    child_initState = ["top_initialState_child_pdgid", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt", "truth_top_initialState_child_e"]
    children_4t = EventCompiler(file_dir, tree, child_initState)
    children_4t.GenerateEvents()

    truth_jets = ["truthjet_flavour", "truthjet_e", "truthjet_phi", "truthjet_eta", "truthjet_pt", "truthjet_flavour"] 
    jet_tF = EventCompiler(file_dir, tree, truth_jets)
    jet_tF.GenerateEvents()

    truth_4t.MatchChildrenParticles(children_4t.EventDictionary)
    truth_4t.EventObjectMap = Map
    truth_4t.MatchToTruthJets(jet_tF.EventDictionary)
    

    #============== Now we test if the matching produces consistent output
    def SafeDict(Particle, Dict):
        try:
            Dict[Particle.PDGID].append(Particle.Mass)
        except KeyError:
            Dict[Particle.PDGID] = []
            Dict[Particle.PDGID].append(Particle.Mass)
        return Dict 

    def SafeDictFlavour(Particle, Dict):
        try:
            Dict[Particle.Flavour].append(Particle.Mass)
        except KeyError:
            Dict[Particle.Flavour] = []
            Dict[Particle.Flavour].append(Particle.Mass)
        return Dict 




    #=== Build the mass of the resonance and the truth tops
    Mass_Resonance_Truth = []
    Mass_Signal_Tops_Truth = []
    Mass_Spectator_Tops_Truth = []
    for i in truth_4t.EventDictionary:
        p_tops = truth_4t.EventDictionary[i]
        
        Z_ = Particle()
        Z_.Name = "Z'"
        for p_t in p_tops:
            if p_t.IsSignal == 1:
                Z_.DecayProducts.append(p_t)
                Mass_Signal_Tops_Truth.append(p_t.Mass)
            else:
                Mass_Spectator_Tops_Truth.append(p_t.Mass)
        
        Z_.ReconstructFourVectorFromProducts()
        Mass_Resonance_Truth.append(Z_.Mass)

    
    #=== Build the mass of the resonance from top decayed children 
    Mass_Resonance_Child = []
    Mass_Signal_Tops_Child = []
    Mass_Spectator_Tops_Child = []
    C_Mass_Signal_Child = {}
    C_Mass_Spectator_Child = {}
    for i in truth_4t.EventDictionary:
        p_tops = truth_4t.EventDictionary[i]
        
        Z_ = Particle()
        Z_.Name = "Z'"
        for p_t in p_tops:
            
            T_ = Particle()
            T_.Name = "t_recon"

            for c_t in p_t.DecayProducts:
                if p_t.IsSignal == 1:
                    T_.DecayProducts.append(c_t)
                    C_Mass_Signal_Child = SafeDict(c_t, C_Mass_Signal_Child) 
                else: 
                    T_.DecayProducts.append(c_t)
                    C_Mass_Spectator_Child = SafeDict(c_t, C_Mass_Spectator_Child)
            
            T_.ReconstructFourVectorFromProducts()
            
            if p_t.IsSignal == 1:
                Mass_Signal_Tops_Child.append(T_.Mass)
                Z_.DecayProducts.append(T_)
            else:
                Mass_Spectator_Tops_Child.append(T_.Mass)
        
        Z_.ReconstructFourVectorFromProducts()
        Mass_Resonance_Child.append(Z_.Mass)
        
    #=== Build the mass of the resonance from top decayed children of children 
    Mass_Resonance_Child_of_Child = []
    Mass_Signal_Tops_Child_of_Child = []
    Mass_Spectator_Tops_Child_of_Child = []

    C_C_Mass_Signal_Child = []
    C_C_Mass_Spectator_Child = []

    Mass_Signal_Child_Child = {}
    Mass_Spectator_Child_Child = {}
    for i in truth_4t.EventDictionary:
        p_tops = truth_4t.EventDictionary[i]
        
        Z_ = Particle()
        Z_.Name = "Z'"
        for p_t in p_tops:
            
            T_ = Particle()
            T_.Name = "t_recon"
            for c_t in p_t.DecayProducts:
                
                D_ = Particle()
                D_.Name = "D_ recon"
                
                for cct in c_t.DecayProducts:
                    D_.DecayProducts.append(cct)
                    
                    if p_t.IsSignal == 1:
                        Mass_Signal_Child_Child = SafeDictFlavour(cct, Mass_Signal_Child_Child)
                    else:
                        Mass_Spectator_Child_Child = SafeDictFlavour(cct, Mass_Spectator_Child_Child)

                # This is done to capture neutrinos and leptons not recorded in the truth_jet
                if len(D_.DecayProducts) == 0:
                    D_.DecayProducts.append(c_t)

                    if p_t.IsSignal == 1:
                        Mass_Signal_Child_Child = SafeDict(c_t, Mass_Signal_Child_Child)
                    else:
                        Mass_Spectator_Child_Child = SafeDict(c_t, Mass_Spectator_Child_Child)
              
                D_.ReconstructFourVectorFromProducts()
                if p_t.IsSignal == 1:
                    C_C_Mass_Signal_Child.append(D_.Mass)
                else:
                    C_C_Mass_Spectator_Child.append(D_.Mass)

                T_.DecayProducts.append(D_)

            
            T_.ReconstructFourVectorFromProducts()
            if p_t.IsSignal == 1:
                Mass_Signal_Tops_Child_of_Child.append(T_.Mass)
                Z_.DecayProducts.append(T_)
            else:
                Mass_Spectator_Tops_Child_of_Child.append(T_.Mass)

        Z_.ReconstructFourVectorFromProducts()
        Mass_Resonance_Child_of_Child.append(Z_.Mass)
    
    Output = {}
    Output["Mass_Resonance_Truth"] = Mass_Resonance_Truth
    Output["Mass_Resonance_Child"] = Mass_Resonance_Child
    Output["Mass_Resonance_Child_of_Child"] = Mass_Resonance_Child_of_Child

    Output["Mass_Signal_Tops_Truth"] = Mass_Signal_Tops_Truth
    Output["Mass_Signal_Tops_Child"] = Mass_Signal_Tops_Child
    Output["Mass_Signal_Tops_Child_of_Child"] = Mass_Signal_Tops_Child_of_Child
    
    Output["Mass_Spectator_Tops_Truth"] = Mass_Spectator_Tops_Truth
    Output["Mass_Spectator_Tops_Child"] = Mass_Spectator_Tops_Child
    Output["Mass_Spectator_Tops_Child_of_Child"] = Mass_Spectator_Tops_Child_of_Child
    
    Output["C_Mass_Signal_Child"] = C_Mass_Signal_Child
    Output["C_C_Mass_Signal_Child"] = C_C_Mass_Signal_Child 
    
    Output["C_Mass_Spectator_Child"] = C_Mass_Spectator_Child
    Output["C_C_Mass_Spectator_Child"] = C_C_Mass_Spectator_Child
    
    Output["Mass_Signal_Child_Child"] = Mass_Signal_Child_Child
    Output["Mass_Spectator_Child_Child"] = Mass_Spectator_Child_Child
    
    return Output
