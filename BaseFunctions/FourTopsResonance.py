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


def AssociateSignalTopsToDetectorJets(file_dir):
    
    # The purpose of this function is to find the jets that belong to the signal tops. Signal being Z' -> t t~
    
    # ===== Get the resonance top information
    TruthTop_Objects = FileObjectsToArrays(ObjectsFromFile(file_dir, "nominal", ["top_FromRes"]))["nominal"]
    TruthJet_Objects = FileObjectsToArrays(ObjectsFromFile(file_dir, "nominal", ["truthjet_flavour"]))["nominal"]
    
    # ===== Child Objects from Branch 
    Child_Objects = ObjectsFromFile(file_dir, "nominal", ["truth_top_child_pdgid", "truth_top_child_e", "truth_top_child_eta", "truth_top_child_phi", "truth_top_child_pt"])[file_dir]

    SignalMass = []
    SpectatorMass = []
    IndividualParticles = []
    ParticlePID = []
    TopMass = []
    for files in Child_Objects:
        ChildPDGID = Child_Objects[files]["nominal"][1]["truth_top_child_pdgid"].array()
        Child_E = Child_Objects[files]["nominal"][1]["truth_top_child_e"].array()
        Child_ETA = Child_Objects[files]["nominal"][1]["truth_top_child_eta"].array()
        Child_PHI = Child_Objects[files]["nominal"][1]["truth_top_child_phi"].array()
        Child_PT = Child_Objects[files]["nominal"][1]["truth_top_child_pt"].array() 

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




