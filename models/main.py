from runner_router import AnalysisBuild, Graphs
from AnalysisG.Events import Event
from AnalysisG import Analysis
from AnalysisG.Tools import Code
import os

mode = "TruthChildren_NoNu" #Jets_Detector"
name = "Example" #"Project_ML"
model = "RPN"
mode_ = 0
gen_data = False

modes = [
    "TruthChildren_NoNu",
#    "TruthJets_NoNu",
#    "Jets_NoNu"
]

params = [
    ("MRK-1" , "ADAM", 1  , {"lr": 1e-6, "weight_decay" : 1e-6},            None,              None),
    ("MRK-2" , "ADAM", 100, {"lr": 1e-6, "weight_decay" : 1e-6},            None,              None),
    ("MRK-3" , "ADAM", 500, {"lr": 1e-6, "weight_decay" : 1e-6},            None,              None),

#    ("MRK-4" , "ADAM", 1  , {"lr": 1e-6, "weight_decay" : 1e-6}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-5" , "ADAM", 100, {"lr": 1e-6, "weight_decay" : 1e-6}, "ExponentialLR", {"gamma"  : 1.0}),
#    ("MRK-6" , "ADAM", 500, {"lr": 1e-6, "weight_decay" : 1e-6}, "ExponentialLR", {"gamma"  : 2.0}),

#    ("MRK-7" , "ADAM", 10 , {"lr": 1e-6, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-8" , "ADAM", 100, {"lr": 1e-6, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-9" , "ADAM", 500, {"lr": 1e-6, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),


#    ("MRK-10", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),
#    ("MRK-11", "SGD",  100, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),
#    ("MRK-12", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),

#    ("MRK-13", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-14", "SGD",  100, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0005}, "ExponentialLR", {"gamma"  : 1.0}),
#    ("MRK-15", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0015}, "ExponentialLR", {"gamma"  : 2.0}),

#    ("MRK-16", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001}, "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-17", "SGD",  100, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0005}, "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-18", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0015}, "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
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

#"RGNN" : RecursiveGraphNeuralNetwork, 
#"BBLR" : BasicBaseLineRecursion,
#"BGGN" : BasicGraphNeuralNetwork,
#"MKGN" : MarkovGraphNet <--- need to fix
auto = AnalysisBuild(name)

for mm in modes:
    if not gen_data: break

    mode = mm
    train_name = "sample-" + mode

    auto.SamplePath = os.environ["Samples"]
    auto.AddDatasetName("ttH-m1000", 1)
    auto.AddDatasetName("ttbar", 1)
    auto.AddDatasetName("tttt (SM)", 1)
    auto.AddDatasetName("ttH", 1)

    auto.Event = Event
    auto.EventCache = True
    auto.MakeEventCache()
    auto.MakeGraphCache(mode)
    auto.QuantizeSamples(100)
    auto.TrainingSample(train_name, 90)

    #auto.EventStop = 10000
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
    Ana.Device = "cuda"
    Ana.TrainingName = "sample-" + mode
    Ana.Model = auto.ModelTrainer(model)
    Ana.BatchSize = batch
#    Ana.kFold = 1
    Ana.Epochs = 100
    Ana.MaxGPU = 20
    Ana.MaxRAM = 200
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
    Ana.ContinueTraining = True
    Ana.DebugMode = False
    Ana.KinematicMap = {"top_edge" : "polar -> N_pT, N_eta, N_phi, N_energy"}
    Ana.Launch()
