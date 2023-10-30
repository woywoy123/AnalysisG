from runner_router import AnalysisBuild, Graphs
from AnalysisG.Events import Event
from AnalysisG import Analysis
from AnalysisG.Tools import Code
import os

mode = "Jets_Detector"
model = "RPN"

#"TruthChildren_All" : GraphChildren,
#"TruthChildren_NoNu" : GraphChildrenNoNu,
#"TruthJets_All" : GraphTruthJet,
#"TruthJets_NoNu": GraphTruthJetNoNu,
#"Jets_All" : GraphJet,
#"Jets_NoNu" : GraphJetNoNu,
#"Jets_Detector" : GraphDetector

# ----- candidates ----- #
#"RPN"  : RecursivePathNetz,


#"RGNN" : RecursiveGraphNeuralNetwork, 
#"BBLR" : BasicBaseLineRecursion,
#"BGGN" : BasicGraphNeuralNetwork,
#"MKGN" : MarkovGraphNet <--- need to fix

auto = AnalysisBuild("Project_ML")
if True:
    auto.SamplePath = os.environ["Samples"]
    #auto.FetchMeta()
    auto.AddDatasetName("ttH-m1000")
    auto.AddDatasetName("ttbar")
    auto.AddDatasetName("tttt (SM)")
    auto.AddDatasetName("ttH")
    auto.Event = Event
    auto.EventCache = True
    auto.MakeEventCache()
    auto.MakeGraphCache(mode)
    auto.QuantizeSamples(20)
    auto.TrainingSample("basic-sample-det", 90)
    for i, job in auto.Make().items():
        job.Threads = 16
        job.Chunks = 10000
        print("-> " + i)
        job.Launch()
        del job

Ana = Analysis()
Ana.ProjectName = "Project_ML"
Ana.Device = "cuda"
Ana.TrainingName = "basic-sample-det"
Ana.Model = auto.ModelTrainer(model)
Ana.kFold = 1
Ana.Epochs = 100
Ana.BatchSize = 1
Ana.MaxGPU = 20
Ana.MaxRAM = 200
Ana.RunName = model + "-" + mode
Ana.Optimizer = "ADAM"
Ana.GraphName = Graphs(mode)._this.__name__
Ana.Tree = "nominal"
Ana.OptimizerParams = {"lr": 1e-6, "weight_decay": 1e-6}
Ana.PlotLearningMetrics = True
Ana.ContinueTraining = True
Ana.DebugMode = False
Ana.KinematicMap = {
                "top_edge" : "polar -> N_pT, N_eta, N_phi, N_energy",
                "res_edge" : "polar -> N_pT, N_eta, N_phi, N_energy"
}
Ana.Launch()
