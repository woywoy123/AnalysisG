from AnalysisG import Analysis 
from AnalysisG.Events import Event
from AnalysisG.Events import GraphChildren, GraphChildrenNoNu


from AnalysisG.Events import GraphTruthJet, GraphDetector
from AnalysisG.Templates import ApplyFeatures
from BasicGraphNeuralNetwork.model import BasicGraphNeuralNetwork
from RecursiveGraphNeuralNetwork.model import RecursiveGraphNeuralNetwork


mode = "TruthChildrenNoNu"
modes = { "TruthJets" : GraphTruthJet, "Jets" : GraphDetector}
modes |= {"TruthChildren" : GraphChildren, "TruthChildrenNoNu" : GraphChildrenNoNu}

Ana = Analysis()
Ana.ProjectName = "Project"
Ana.InputSample("bsm-1000", "/home/tnom6927/Downloads/samples/Dilepton_/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
Ana.InputSample("ttbar", "/home/tnom6927/Downloads/samples/Dilepton/ttbar/DAOD_TOPQ1.27296255._000017.root")
Ana.Event = Event 

Ana.EventGraph = modes[mode]
Ana.DataCache = True
Ana.EventStop = 1000


ApplyFeatures(Ana)
Ana.TrainingSampleName = mode
Ana.Device = "cuda"
Ana.kFolds = 1
Ana.Epochs = 10000
Ana.BatchSize = 100
Ana.ContinueTraining = False
Ana.Optimizer = "ADAM" 
Ana.OptimizerParams = {"lr" : 1e-4, "weight_decay" : 1e-3}
Ana.Model = RecursiveGraphNeuralNetwork
Ana.DebugMode = True
Ana.EnableReconstruction = False
Ana.PurgeCache = False
Ana.Launch
