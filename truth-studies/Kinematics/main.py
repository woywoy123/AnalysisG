from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from TruthTops import *
from TopChildren import *
from TruthJetMatching import * 

direc = "/CERN/Samples/SingleLepton/Collections/ttH_tttt_m1000"
Ana = Analysis()
Ana.InputSample("tttt", direc)
Ana.Event = Event
Ana.EventStop = 100
Ana.chnk = 100
Ana.EventCache = True
Ana.DumpPickle = True
Ana.Launch()

# ------ Top Centric Plots ----- # 
# Figures 1.1: "a" and "b"
#ResonanceDecayModes(Ana)

# Figures 1.1: "c"
#ResonanceMassFromTops(Ana)

# Figures 1.1: "d"
#DeltaRTops(Ana)

# Figures 1.1: "e" and "f"
#TopKinematics(Ana)

# ------ Top Child Centric Plots ----- # 
# Figures 2.1:  "a"
#TopChildrenPDGID(Ana)

# Figures 2.1: "a"
#TopChildrenPDGID(Ana)

# Figures 2.1: "b" and "c"
#ReconstructedMassFromChildren(Ana)

# Figures 2.1: "d", "e", "f", "g" 
#DeltaRChildren(Ana)

# Figures 2.1: "g"
#FractionPTChildren(Ana)

# ------ Truth Jet Centric Plots ----- # 
# Figures 3.1: "a" - "d"
TruthJetPartons(Ana)

# Figures 3.1: 
PartonToChildTruthJet(Ana)


#TruthTopsAll(Ana)
#TruthTopsHadron(Ana)
#TruthChildrenAll(Ana)
#TruthChildrenHadron(Ana)
#TruthJetAll(Ana)
