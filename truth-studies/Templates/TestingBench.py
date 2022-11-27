from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Events import EventGraphTruthTops
from AnalysisTopGNN.Events import EventGraphTruthTopChildren
from AnalysisTopGNN.Events import EventGraphTruthJetLepton
from EventFeatureTemplate import ApplyFeatures


direc = "/CERN/Samples/Dilepton/Collections/MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000/"

Ana = Analysis()
Ana.Event = Event
Ana.EventCache = True 
Ana.DumpPickle = True
Ana.chnk = 100
Ana.EventStop = 50
Ana.InputSample("bsm4-tops", direc)
Ana.Launch()

GR = Analysis()
GR.InputSample("bsm4-tops")
GR.EventCache = True
GR.EventGraph = EventGraphTruthTopChildren
ApplyFeatures(GR, "TruthTops")
GR.TestFeatures(5)

GR = Analysis()
GR.InputSample("bsm4-tops")
GR.EventCache = True
GR.EventGraph = EventGraphTruthTopChildren
ApplyFeatures(GR, "TruthChildren")
GR.TestFeatures(5)

GR = Analysis()
GR.InputSample("bsm4-tops")
GR.EventCache = True
GR.EventGraph = EventGraphTruthJetLepton
ApplyFeatures(GR, "TruthJets")
GR.TestFeatures(5)

