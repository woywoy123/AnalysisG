from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Events import EventGraphTruthTopChildren
from AnalysisTopGNN.Features import ApplyFeatures
from BasicBaseLineModel.V3.BasicBaseLine import BasicBaseLineRecursion

Ana = Analysis()
Ana.InputSample("bsm-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
Ana.Event = Event 
Ana.EventCache = True 

Ana.EventGraph = EventGraphTruthTopChildren
Ana.DataCache = True 
Ana.EventStop = 100

Ana.DumpPickle = True 
ApplyFeatures(Ana, "TruthChildren")
Ana.TrainingSampleName = "Children"
#Ana.Device = "cuda"
#Ana.kFolds = 10
#Ana.Epochs = 10
#Ana.BatchSize = 2
#Ana.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
#Ana.Model = BasicBaseLineRecursion()
#Ana.DebugMode = True
Ana.Launch()
