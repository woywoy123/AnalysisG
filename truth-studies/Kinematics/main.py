from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
#from TruthTops import *
#from TruthChildren import *
#from TruthMatching import *
from TruthTops_ import *
from TopChildren_ import *

direc = "/CERN/Samples/Processed/bsm4tops/mc16e"
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

# ------ Top Centric Plots ----- # 
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






#TruthTopsAll(Ana)
#TruthTopsHadron(Ana)
#TruthChildrenAll(Ana)
#TruthChildrenHadron(Ana)
#TruthJetAll(Ana)
