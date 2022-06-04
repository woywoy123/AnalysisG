from PresentationPlots.GenericFunctions import *
from Functions.Particles.Particles import Particle

Out = "PresentationPlots/Presentation1/Plots/"

def TopMassAnalysis(ev):
    Names = ["TruthTopMass", "TopMassFromChildren", "TopMassFromChildrenInit"]
    Backup = {i : [] for i in Names}

    for i in ev.Events:
        event = ev.Events[i]["nominal"]

        for t in event.TruthTops:
            t.CalculateMass()
            Backup["TruthTopMass"].append(t.Mass_GeV)
            t.CalculateMassFromChildren()
            Backup["TopMassFromChildrenInit"].append(t.Mass_init_GeV)
        
        for t in event.TopPostFSR:
            t.CalculateMassFromChildren()
            Backup["TopMassFromChildren"].append(t.Mass_init_GeV)

    Histograms_Template("Invariant Mass of Monte Carlo Truth Top", "Mass (GeV)", "Entries", 
            250, 172.48, 172.52, Data = Backup["TruthTopMass"], 
            FileName = "TopMassTruth", Dir = Out + "TopMass", Alpha = 1, DPI = 200)
    
    Histograms_Template("Reconstructed Monte Carlo Truth Top from Children (Pre-FSR)", "Mass (GeV)", "Entries", 
            250, 150, 200, Data = Backup["TopMassFromChildrenInit"], 
            FileName = "TopMassChildrenPreFSR", Dir = Out + "TopMass", Alpha = 1, DPI = 200)

    Histograms_Template("Reconstructed Monte Carlo Truth Top from Children (Post-FSR)", "Mass (GeV)", "Entries", 
            250, 20, 200, Data = Backup["TopMassFromChildrenInit"], 
            FileName = "TopMassChildrenPostFSR", Dir = Out + "TopMass", Alpha = 1, DPI = 200)
   

def EdgeAnalysisOfChildren(ev):
    for i in ev.Events:
        event = ev.Events[i]

        for ch in event.TopPostFSRChildren:
            print(ch.Index)




def CreatePlots(FileDir, CreateCache):
    CreateCache = False
    ev = CreateWorkspace("Presentation1", FileDir, CreateCache, 100)
    #TopMassAnalysis(ev)
    EdgeAnalysisOfChildren()
    
    return True
