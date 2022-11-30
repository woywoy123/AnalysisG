from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from TruthTops import *
from TopChildren import *
from TruthJetMatching import * 
import os
import shutil

massPoints = ["400", "500", "600", "700", "800", "900", "1000"]
Modes = ["SingleLepton", "Dilepton"]

for Mode in Modes:
    for massPoint in massPoints:
        direc = "/CERN/Samples/" + Mode + "/Collections/ttH_tttt_m" + massPoint
        Ana = Analysis()
        Ana.InputSample("tttt", direc)
        Ana.Event = Event
        #Ana.EventStop = 100
        Ana.chnk = 100
        Ana.EventCache = True
        Ana.DumpPickle = True
        Ana.Launch()
        
        # ------ Top Centric Plots ----- # 
        # Figures 1.1: "a" and "b"
        ResonanceDecayModes(Ana)
        
        # Figures 1.1: "c"
        ResonanceMassFromTops(Ana)
        
        # Figures 1.1: "d"
        DeltaRTops(Ana)
        
        # Figures 1.1: "e" and "f"
        TopKinematics(Ana)
        
        # ------ Top Child Centric Plots ----- # 
        # Figures 2.1:  "a"
        TopChildrenPDGID(Ana)
        
        # Figures 2.1: "a"
        TopChildrenPDGID(Ana)
        
        # Figures 2.1: "b" and "c"
        ReconstructedMassFromChildren(Ana)
        
        # Figures 2.1: "d", "e", "f", "g" 
        DeltaRChildren(Ana)
        
        # Figures 2.1: "g"
        FractionPTChildren(Ana)
        
        # ------ Truth Jet Centric Plots ----- # 
        # Figures 3.1: "a" - "d"
        TruthJetPartons(Ana)
        
        # Figures 3.1: "e"
        PartonToChildTruthJet(Ana)
        
        # Figures 3.1: "f" and "g"
        eff = ReconstructedTopMassTruthJet(Ana)
        f = open(Mode + "_" + massPoint + ".txt", "w")
        f.write("\n".join([i + "-" + str(eff[i]) for i in eff]))
        f.close()
    
        os.makedirs("./" + Mode + "/" + massPoint, exist_ok = True)
        shutil.move("./Figures", "./" + Mode + "/" + massPoint)
        exit()
