from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Events import EventGraphTruthTopChildren
from AnalysisTopGNN.Features import ApplyFeatures

Ana = Analysis()
Ana.InputSample("bsm-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
#Ana.Event = Event 
#Ana.EventCache = True 

Ana.EventGraph = EventGraphTruthTopChildren
Ana.DataCache = True 
Ana.EventStop = 100

Ana.DumpPickle = True 
ApplyFeatures(Ana, "TruthChildren")
Ana.Launch()
