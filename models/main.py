from runner_router import AnalysisBuild, Graphs
from AnalysisG.Submission import Condor
from AnalysisG.Events import Event
from AnalysisG.Tools import Code
from AnalysisG import Analysis
import os


mode_ = 0
device = "cuda:1"
name = "ModelTraining"
model = "GNNEXP" #"RNN" #"RMGN"
gen_data = False

modes = [
#    "TruthChildren_All",
#    "TruthChildren_NoNu",
#    "TruthJets_NoNu",
    "TruthJets_All",
#    "Jets_NoNu"
]

params = [
#    ("MRK-1" , "ADAM", 1 , {"lr": 1e-3},            None,              None),
#    ("MRK-2" , "ADAM", 10, {"lr": 1e-3},            None,              None),
#    ("MRK-3" , "ADAM", 50, {"lr": 1e-3},            None,              None),

#    ("MRK-4" , "ADAM", 1 , {"lr": 1e-3, "weight_decay" : 1e-3}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-5" , "ADAM", 10, {"lr": 1e-3, "weight_decay" : 1e-1}, "ExponentialLR", {"gamma"  : 0.7}),
#    ("MRK-6" , "ADAM", 50, {"lr": 1e-3, "weight_decay" : 1e1 }, "ExponentialLR", {"gamma"  : 0.9}),

#    ("MRK-7" , "SGD", 1 , {"lr": 1e-3, "weight_decay" : 1e-3},      "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e0}),
#    ("MRK-8" , "SGD", 10, {"lr": 1e-3, "weight_decay" : 1e-3},      "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e1}),
#    ("MRK-9" , "SGD", 50, {"lr": 1e-3, "weight_decay" : 1e-3},      "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e1}),


#    ("MRK-10", "SGD", 1 , {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0001},            None,              None),
#    ("MRK-11", "SGD", 10, {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0001},            None,              None),
#    ("MRK-12", "SGD", 50, {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0001},            None,              None),

#    ("MRK-13", "SGD", 1 , {"lr": 1e-3, "weight_decay" : 1e-1, "momentum" : 0.0001}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-14", "SGD", 10, {"lr": 1e-3, "weight_decay" : 1e-1, "momentum" : 0.0005}, "ExponentialLR", {"gamma"  : 1.0}),
#    ("MRK-15", "SGD", 50, {"lr": 1e-3, "weight_decay" : 1e-1, "momentum" : 0.0015}, "ExponentialLR", {"gamma"  : 2.0}),

    ("MRK-16", "SGD", 1 , {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0001}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e-1}),
    ("MRK-17", "SGD", 10, {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0005}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e0 }),
    ("MRK-18", "SGD", 50, {"lr": 1e-3, "weight_decay" : 1e-3, "momentum" : 0.0015}, "CyclicLR", {"base_lr" : 1e-3, "max_lr" : 1e1 }),
]


#"TruthChildren_All" : GraphChildren,
#"TruthChildren_NoNu" : GraphChildrenNoNu,
#"TruthJets_All" : GraphTruthJet,
#"TruthJets_NoNu": GraphTruthJetNoNu,
#"Jets_All" : GraphJet,
#"Jets_NoNu" : GraphJetNoNu,
#"Jets_Detector" : GraphDetector

# ----- candidates ----- #
#"RPN"  : RecursivePathNetz,
#"RNN"  : RecuriveNuNetz
#"RMGN" : RecursiveMarkovianGraphNet

trig = True
auto = AnalysisBuild(name)
for mm in modes:
    if not gen_data: break
    mode = mm
    train_name = "sample-" + mode

    if trig:
        auto.SamplePath = os.environ["Samples"]
        auto.AddDatasetName("ttH-m1000", 2)
        auto.AddDatasetName("ttbar", 2)
        auto.AddDatasetName("tttt (SM)", 2)
        auto.AddDatasetName("ttH", 2)
        #auto.AddDatasetName("3-top", 2)
        #auto.AddDatasetName("VH", 2)
        #auto.AddDatasetName("ttll", 2)
        #auto.AddDatasetName("ttW", 2)

        auto.Event = Event
        auto.EventCache = True
        auto.MakeEventCache()
        trig = False

    auto.MakeGraphCache(mode)
    auto.QuantizeSamples(10)
    auto.TrainingSample(train_name, 90)

    for i, job in auto.Make().items():
        job.Threads = 22
        job.Chunks = 1000
        print("-> " + i)
        job.Launch()
        del job

mode = modes[mode_]
for this in params:
    run_, min_, batch, min_param, sch_, sch_param = this

    Ana = Analysis()
    Ana.ProjectName = name
    Ana.Device = device
    Ana.TrainingName = "sample-" + mode
    Ana.Model = auto.ModelTrainer(model)
    Ana.ModelParams = {"device" : device}
    Ana.BatchSize = batch
    Ana.kFold = 1
    Ana.Epochs = 200
    Ana.MaxGPU = 20
    Ana.MaxRAM = 50
    Ana.Tree = "nominal"
    Ana.EventName = None
    Ana.RunName = run_
    Ana.GraphName = Graphs(mode)._this.__name__

    Ana.Optimizer = min_
    Ana.OptimizerParams = min_param
    if sch_ is not None:
        Ana.Scheduler = sch_
        Ana.SchedulerParams = sch_param

    Ana.PlotLearningMetrics = True
    Ana.ContinueTraining = False
    Ana.DebugMode = False
    Ana.KinematicMap = {"top_edge" : "polar -> N_pT, N_eta, N_phi, N_energy"}
    Ana.Launch()
