from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from TruthTop.GraphFeature import *






direc = "/CERN/Samples/Dilepton/Collections/MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000/"

#Ana = Analysis()
#Ana.Event = Event
#Ana.EventCache = True 
#Ana.DumpPickle = True
#Ana.chnk = 1000
#Ana.EventStop = 100
#Ana.InputSample("bsm4-tops", direc)
#Ana.Launch()

GR = Analysis()
GR.EventCache = True
GR.EventGraph = EventGraphTruthTopChildren
GR.AddGraphFeature(SignalEvent, "SignalEvent")
GR.InputSample("bsm4-tops")
GR.TestFeatures(1000)

