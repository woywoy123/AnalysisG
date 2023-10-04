from AnalysisG import Analysis
from AnalysisG.Events import Event
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Events import GraphChildren, GraphChildrenNoNu
from AnalysisG.Events import GraphTruthJet, GraphDetector
import os

mode = "TruthChildrenNoNu"
modes = {
            "TruthChildrenNoNu" : GraphChildrenNoNu, 
            "GraphChildren" : GraphChildren, 
            "GraphTruthJets" : GraphTruthJet,
            "GraphDetector" : GraphDetector
}

ttbar = [
    "DAOD_TOPQ1.25521412._000370.root",
]

bsm4t = [
    "DAOD_TOPQ1.21955717._000007.root",
]

pth = os.environ["Samples"]

Ana = Analysis()
Ana.ProjectName = "Project_ML"
Ana.InputSample("bsm-1000", {pth + "/ttZ_1000/" : bsm4t})
Ana.InputSample("ttbar", {pth + "/Dilepton/ttbar/" : ttbar})
Ana.Event = Event
Ana.Graph = modes[mode]
ApplyFeatures(Ana, mode)
Ana.EventCache = True
Ana.Launch()


Ana = Analysis()
Ana.ProjectName = "Project"
Ana.InputSample("bsm-1000")
Ana.InputSample("ttbar")
Ana.DataCache = True
Ana.kFolds = 10
Ana.TrainingSize = 90
Ana.TrainingName = "basic-sample"
Ana.Launch()

#Ana = Analysis()
#Ana.ProjectName = "Project"
#Ana.TrainingName = "k10"
#Ana.Device = "cuda"
#Ana.kFold = 1
#Ana.Epochs = 100
#Ana.BatchSize = 20
#Ana.ContinueTraining = False
#Ana.Optimizer = "ADAM"
#Ana.OptimizerParams = {"lr": 1e-3, "weight_decay": 1e-3}
#Ana.Model = RecursiveGraphNeuralNetwork
#Ana.DebugMode = False
#Ana.EnableReconstruction = False
#Ana.Launch
