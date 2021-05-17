from BaseFunctions.IO import *
from BaseFunctions.UpRootFunctions import *
from BaseFunctions.Physics import *
import ROOT

def ReadLeafsFromResonance(file_dir):

    Entry_Objects = ObjectsFromFile(file_dir, "nominal", ["top_FromRes", "truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e"])
    skimmed = FileObjectsToArrays(Entry_Objects)
    branch_map = skimmed["nominal"]

    Simple_TopMass = []
    Resonance_Mass = []

    for i in range(len(branch_map["top_FromRes"])):
        res_v = branch_map["top_FromRes"][i]
        pt_v = branch_map["truth_top_pt"][i]
        eta_v = branch_map["truth_top_eta"][i]
        phi_v = branch_map["truth_top_phi"][i]
        e_v = branch_map["truth_top_e"][i]
       
        lor_pair = []
        # Create a vector element for each top
        for x in range(len(res_v)):
            v = ROOT.Math.PtEtaPhiEVector()

            if res_v[x] != 0:
                lor_pair.append(v.SetCoordinates(pt_v[x], eta_v[x], phi_v[x], e_v[x]))
            else: 
                Simple_TopMass.append(v.SetCoordinates(pt_v[x], eta_v[x], phi_v[x], e_v[x]).mass())

            # Collect the individual top masses 
            Simple_TopMass.append(float(v.mass()) / 1000)

        # Get the mass of the resonance   
        if len(lor_pair) != 2:
            continue
        delta = lor_pair[0] + lor_pair[1]
        Resonance_Mass.append(float(delta.mass()) / 1000)
       
    return Simple_TopMass, Resonance_Mass

def AssociateSignalTopsToDetectorJets(file_dir):
    
    # The purpose of this function is to find the jets that belong to the signal tops. Signal being Z' -> t t~
    
    # ===== Get the resonance top information
    TruthTop_Objects = FileObjectsToArrays(ObjectsFromFile(file_dir, "nominal", ["top_FromRes"]))["nominal"]
    TruthJet_Objects = FileObjectsToArrays(ObjectsFromFile(file_dir, "nominal", ["truthjet_flavour"]))["nominal"]
    
    # ===== Child Objects from Branch 
    Child_Objects = ObjectsFromFile(file_dir, "nominal", ["top_initialState_child_pdgid", "truth_top_initialState_child_e", "truth_top_initialState_child_eta", "truth_top_initialState_child_phi", "truth_top_initialState_child_pt"])[file_dir]

    SignalMass = []
    SpectatorMass = []
    IndividualParticles = []
    ParticlePID = []
    TopMass = []
    for files in Child_Objects:
        ChildPDGID = Child_Objects[files]["nominal"][1]["top_initialState_child_pdgid"].array()
        Child_E = Child_Objects[files]["nominal"][1]["truth_top_initialState_child_e"].array()
        Child_ETA = Child_Objects[files]["nominal"][1]["truth_top_initialState_child_eta"].array()
        Child_PHI = Child_Objects[files]["nominal"][1]["truth_top_initialState_child_phi"].array()
        Child_PT = Child_Objects[files]["nominal"][1]["truth_top_initialState_child_pt"].array() 

        FromRes = TruthTop_Objects["top_FromRes"]
        
        for e in range(len(ChildPDGID)):
            sig_t = []
            spec_t = []
            
            for t in range(len(FromRes[e])):
                if FromRes[e][t] == 1:
                    sig_t.append(MultiParticleVector(Child_PT[e][t], Child_ETA[e][t], Child_PHI[e][t], Child_E[e][t]))
                    for v in range(len(Child_PT[e][t])):
                        IndividualParticles.append(ParticleVector(Child_PT[e][t][v], Child_ETA[e][t][v], Child_PHI[e][t][v], Child_E[e][t][v]).mass())
                        #if abs(ChildPDGID[e][t][v]) == 3 or abs(ChildPDGID[e][t][v]) == 4:
                        #    print(ParticleVector(Child_PT[e][t][v], Child_ETA[e][t][v], Child_PHI[e][t][v], Child_E[e][t][v]).M(), ChildPDGID[e][t][v])

                        ParticlePID.append(ChildPDGID[e][t][v])
                else:
                    spec_t.append(MultiParticleVector(Child_PT[e][t], Child_ETA[e][t], Child_PHI[e][t], Child_E[e][t]))
                
                TopMass.append(MultiParticleVector(Child_PT[e][t], Child_ETA[e][t], Child_PHI[e][t], Child_E[e][t]).mass() / 1000)

            if (len(sig_t) != 2):
                continue

            delta = sig_t[0] + sig_t[1]
            SignalMass.append(delta.M()/ 1000)

    return SignalMass, IndividualParticles, ParticlePID, TopMass




