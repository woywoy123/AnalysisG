from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Submission import Condor
from AnalysisG.Events import Event
from runner_router import AnalysisBuild, Graphs
from AnalysisG import Analysis
import os


this_ev = Event
mode = "Jets_All"
model = "GNNEXP"
smple = os.environ["Samples"]
folds = 10
venv = "/nfs/dust/atlas/user/woywoy12/AnalysisG/setup-scripts/source_this.sh"


params = [
    ("MRK-1" , "ADAM", 1  , {"lr": 1e-3, "weight_decay" : 1e-3},            None,              None),
    ("MRK-2" , "ADAM", 1  , {"lr": 1e-3, "weight_decay" : 1e-3},            None,              None),
    ("MRK-3" , "ADAM", 500, {"lr": 1e-3, "weight_decay" : 1e-1},            None,              None),
#
#    ("MRK-4" , "ADAM", 1  , {"lr": 1e-3, "weight_decay" : 1e-3}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-5" , "ADAM", 100, {"lr": 1e-3, "weight_decay" : 1e-6}, "ExponentialLR", {"gamma"  : 0.7}),
#    ("MRK-6" , "ADAM", 500, {"lr": 1e-3, "weight_decay" : 1e-6}, "ExponentialLR", {"gamma"  : 0.9}),
#
#    ("MRK-7" , "SGD", 1  , {"lr": 1e-6, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-8" , "SGD", 100, {"lr": 1e-3, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-9" , "SGD", 500, {"lr": 1e-6, "weight_decay" : 1e-6},      "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#
#    ("MRK-10", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),
#    ("MRK-11", "SGD",  100, {"lr": 1e-3, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),
#    ("MRK-12", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001},            None,              None),
#
#    ("MRK-13", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001}, "ExponentialLR", {"gamma"  : 0.5}),
#    ("MRK-14", "SGD",  100, {"lr": 1e-3, "weight_decay" : 1e-6, "momentum" : 0.0005}, "ExponentialLR", {"gamma"  : 1.0}),
#    ("MRK-15", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0015}, "ExponentialLR", {"gamma"  : 2.0}),
#
#    ("MRK-16", "SGD",  1  , {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0001}, "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
#    ("MRK-17", "SGD",  100, {"lr": 1e-3, "weight_decay" : 1e-6, "momentum" : 0.0005}, "CyclicLR", {"base_lr" : 1e-6, "max_lr" : 1e-1}),
#    ("MRK-18", "SGD",  500, {"lr": 1e-6, "weight_decay" : 1e-6, "momentum" : 0.0015}, "CyclicLR", {"base_lr" : 1e-9, "max_lr" : 1e-4}),
]




def EventGen(name, path):
    ana = Analysis()
    ana.InputSample(name, path)
    ana.EventCache = True
    ana.Event = this_ev
    ana.Chunks = 1000
    ana.Threads = 12
    return ana

def GraphGen(name):
    tmp = this_ev()
    ana = Analysis()
    ana.DataCache = True
    ana.EventName = tmp.__name__()
    ana.Graph = Graphs(mode)._this
    ana.InputSample(name)
    ApplyFeatures(ana, mode.split("_")[0])
    ana.Chunks = 1000
    ana.Threads = 12
    return ana

def TrainGen(names, train_name):
    ana = Analysis()
    ana.TrainingName = train_name
    for n in names: ana.InputSample(n)
    ana.TrainingSize = 40
    ana.DataCache = True
    ana.kFolds = folds
    ana.Threads = 12
    ana.GraphName = Graphs(mode)._this.__name__
    return ana

def OptimGen(this, fold, train_name):
    run_, min_, batch, min_param, sch_, sch_param = this

    auto = AnalysisBuild("tmp")
    Ana = Analysis()
    Ana.Device = "cuda"
    Ana.TrainingName = train_name
    Ana.Model = auto.ModelTrainer(model)
    Ana.BatchSize = batch
    Ana.kFold = fold
    Ana.Epochs = 100
    Ana.Threads = 12
    Ana.MaxGPU = 4
    Ana.MaxRAM = 32
    Ana.Tree = "nominal"
    Ana.OpSysVer = '"CentOS7"'
    Ana.EventName = None
    Ana.DataCache = True
    Ana.RunName = run_
    Ana.GraphName = Graphs(mode)._this.__name__

    Ana.Optimizer = min_
    Ana.OptimizerParams = min_param
    if sch_ is not None:
        Ana.Scheduler = sch_
        Ana.SchedulerParams = sch_param

    Ana.PlotLearningMetrics = False
    Ana.ContinueTraining = False
    Ana.DebugMode = False
    Ana.KinematicMap = {"top_edge" : "polar -> N_pT, N_eta, N_phi, N_energy"}
    return Ana


con = Condor()
con.PythonVenv = venv
con.ProjectName = "ModelTrainer"
samples_map = con.lsFiles(smple, ".root")
samples_map = [i for i in samples_map if "ttH" in i][:4]
samples_map = {i.split("/")[-1].replace(".root", "") : i for i in samples_map}
event_ = []
for name in samples_map:
    con.AddJob(name + "-ev", EventGen(name, samples_map[name]), memory = "32GB", time = "12hrs")
    event_.append(name + "-ev")

graphs_ = []
for name in samples_map:
    con.AddJob(name + "-gr", GraphGen(name), memory = "32GB", time = "12h", waitfor = event_)
    graphs_.append(name + "-gr")

con.AddJob(mode + "train", TrainGen(samples_map, mode), memory = "32GB", time = "48h", waitfor = graphs_)

for i in params:
    for k in range(folds):
        con.AddJob("k-"+ str(k) + "-" + i[0], OptimGen(i, k+1, mode), waitfor = [mode + "train"], memory = "32GB", time = "48h")
#con.LocalRun()
con.SubmitToCondor()
