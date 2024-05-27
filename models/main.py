from runner_router import AnalysisBuild, Graphs
from AnalysisG.Submission import Condor
from AnalysisG.Events import Event
from AnalysisG.Tools import Code
from AnalysisG import Analysis
import os


mode_ = 0
path = "./"
device = "cuda:0"
name = "ModelTrainingSmall"
model = "GRNN" #"RNN" #"RMGN"
gen_data  = False
trigEvent = False
trigGraph = False

# ----- candidates ----- #
# GRNN: RecursiveGraphNeuralNetwork

#"TruthChildren_All"  : GraphChildren,
#"TruthChildren_NoNu" : GraphChildrenNoNu,
#"TruthJets_All"      : GraphTruthJet,
#"TruthJets_NoNu"     : GraphTruthJetNoNu,
#"Jets_All"           : GraphJet,
#"Jets_NoNu"          : GraphJetNoNu,
#"Jets_Detector"      : GraphDetector

modes = [
#    "TruthChildren_All",
#    "TruthChildren_NoNu",
    "TruthJets_All",
#    "TruthJets_NoNu",
#    "Jets_All",
#    "Jets_Detector"
]

params = [
    ("MRK-1" , "ADAM", 1 , {"lr": 1e-3}, None, None),
#    ("MRK-2" , "ADAM", 1 , {"lr": 1e-4}, None, None),
#    ("MRK-3" , "ADAM", 1 , {"lr": 1e-5}, None, None),

#    ("MRK-4" , "ADAM", 1 , {"lr": 1e-3}, "ExponentialLR", {"gamma" : 0.5}),
#    ("MRK-5" , "ADAM", 1 , {"lr": 1e-4}, "ExponentialLR", {"gamma" : 0.7}),
#    ("MRK-6" , "ADAM", 1 , {"lr": 1e-5}, "ExponentialLR", {"gamma" : 0.9}),

#    ("MRK-7" , "SGD", 1 , {"lr": 1e-3}, "CyclicLR", {"base_lr" : 1e-6, "max_lr" : 1e-1}),
#    ("MRK-8" , "SGD", 1 , {"lr": 1e-4}, "CyclicLR", {"base_lr" : 1e-6, "max_lr" : 1e-1}),
#    ("MRK-9" , "SGD", 1 , {"lr": 1e-5}, "CyclicLR", {"base_lr" : 1e-6, "max_lr" : 1e-1}),

#    ("MRK-10", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0001}, None, None),
#    ("MRK-11", "SGD", 1 , {"lr": 1e-4, "momentum" : 0.0001}, None, None),
#    ("MRK-12", "SGD", 1 , {"lr": 1e-5, "momentum" : 0.0001}, None, None),

#    ("MRK-13", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0001}, "ExponentialLR", {"gamma" : 0.5}),
#    ("MRK-14", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0005}, "ExponentialLR", {"gamma" : 1.0}),
#    ("MRK-15", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0015}, "ExponentialLR", {"gamma" : 2.0}),

#    ("MRK-16", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0001}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e-1}),
#    ("MRK-17", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0005}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e0 }),
#    ("MRK-18", "SGD", 1 , {"lr": 1e-3, "momentum" : 0.0015}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e1 }),
]


auto = AnalysisBuild(name)
auto.OutputDir = path
auto.SamplePath = path + "Dilepton" #os.environ["Samples"]
auto.Event = Event

for mm in modes:
    if not gen_data: break
    train_name = "model-train" #"sample-" + mm
    if trigEvent:
        auto.AddSampleNameEvent("other")
        auto.AddSampleNameEvent("t")
        auto.AddSampleNameEvent("tt")
        auto.AddSampleNameEvent("ttbar")
        auto.AddSampleNameEvent("V")
        auto.AddSampleNameEvent("Vll")
        auto.AddSampleNameEvent("Vqq")

        auto.AddSampleNameEvent("ttH")
        auto.AddSampleNameEvent("tttt", 4)
        auto.AddSampleNameEvent("ttX")
        auto.AddSampleNameEvent("ttXll")
        auto.AddSampleNameEvent("ttXqq")
        auto.AddSampleNameEvent("ttZ-1000")
        trigEvent = True

    if trigGraph:
        auto.AddSampleNameGraph(mm, "other")
        auto.AddSampleNameGraph(mm, "t")
        auto.AddSampleNameGraph(mm, "tt")
        auto.AddSampleNameGraph(mm, "ttbar")
        auto.AddSampleNameGraph(mm, "V")
        auto.AddSampleNameGraph(mm, "Vll")
        auto.AddSampleNameGraph(mm, "Vqq")

        auto.AddSampleNameGraph(mm, "ttH")
        auto.AddSampleNameGraph(mm, "tttt")
        auto.AddSampleNameGraph(mm, "ttX")
        auto.AddSampleNameGraph(mm, "ttXll")
        auto.AddSampleNameGraph(mm, "ttXqq")
        auto.AddSampleNameGraph(mm, "ttZ-1000")

    if trigEvent:
        auto.TrainingSample(mm, train_name, 50)
        trigEvent = False

mode = modes[mode_]
for this in params:
    run_, min_, batch, min_param, sch_, sch_param = this

    ana = Analysis()
    ana.ProjectName = name
    ana.Device = device
    ana.OutputDirectory = path
    ana.TrainingName = "model-train"
    ana.Model = auto.ModelTrainer(model)
    ana.ModelParams = {"device" : device}
    ana.BatchSize = batch
    ana.kFold   = 1
    ana.Epochs  = 200
    ana.MaxGPU  = 20
    ana.MaxRAM  = 200
    ana.Threads = 6
    ana.Tree = "nominal"
    ana.EventName = None
    ana.RunName = run_ + "-" + mode
    ana.GraphName = Graphs(mode)._this.__name__

    ana.Optimizer = min_
    ana.OptimizerParams = min_param
    if sch_ is not None:
        ana.Scheduler = sch_
        ana.SchedulerParams = sch_param

    ana.PlotLearningMetrics = True
    ana.ContinueTraining = True
    ana.DebugMode = False
    ana.KinematicMap = {"top_edge" : "polar -> N_pT, N_eta, N_phi, N_energy"}
    ana.ModelInjection = False
    ana.Launch()
