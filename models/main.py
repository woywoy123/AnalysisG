from AnalysisG import Analysis 
from AnalysisG.Events import Event
from AnalysisG.Events import GraphChildren, GraphDetector
from AnalysisG.Templates import ApplyFeatures
from BasicGraphNeuralNetwork.model import BasicGraphNeuralNetwork


Ana = Analysis()
Ana.ProjectName = "Project"
Ana.InputSample("bsm-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
Ana.InputSample("ttbar", "/home/tnom6927/Downloads/samples/ttbar/DAOD_TOPQ1.27296255._000017.root")
Ana.Event = Event 

Ana.EventGraph = GraphChildren
Ana.DataCache = True
Ana.EventStop = 1000

ApplyFeatures(Ana, "TruthChildren")
Ana.TrainingSampleName = "Jets"
Ana.Device = "cuda"
Ana.kFolds = 10
Ana.Epochs = 100
Ana.BatchSize = 10
Ana.ContinueTraining = False
Ana.Optimizer = "ADAM" 
Ana.OptimizerParams = {"lr" : 1e-4, "weight_decay" : 1e-3}
Ana.Model = BasicGraphNeuralNetwork
Ana.DebugMode = False
Ana.EnableReconstruction = True
Ana.PurgeCache = False
Ana.Launch
