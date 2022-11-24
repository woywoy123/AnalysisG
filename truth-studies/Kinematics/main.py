from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from TruthTops import *
from TruthChildren import *
from TruthMatching import *

#direc = "/CERN/Samples/Processed/bsm4tops/mc16a"
Ana = Analysis()
Ana.InputSample("tttt", direc)
Ana.Event = Event
#Ana.EventStop = 100
Ana.chnk = 100
Ana.EventCache = True
Ana.DumpPickle = True
Ana.Launch()

TruthTopsAll(Ana)
#TruthTopsHadron(Ana)
#TruthChildrenAll(Ana)
#TruthChildrenHadron(Ana)
#TruthJetAll(Ana)
