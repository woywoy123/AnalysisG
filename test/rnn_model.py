#import torch
import AnalysisG
from AnalysisG import Analysis
from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events import SSML_MC20
from AnalysisG.models import *

root1 = "./samples/dilepton/DAOD_TOPQ1.21955717._000001.root"
#root1 = "/home/tnom6927/Downloads/mc16/tmp/*" #user.tnommens.40945479._000001.output.root"
#root1 = "/home/tnom6927/Downloads/mc16/ttH_tttt_m400/*"

x = BSM4Tops()
#x = SSML_MC20()
#tt = GraphChildren()
#tt = GraphTruthJets()
#tt = GraphJets()
#tt = GraphDetectorLep() 

tt = GraphDetector()
tt.PreSelection = True

params = [
            ("MRK-1", "adam", {"lr" : 1e-4}, "steplr", {"gamma" : 0.9, "step_size" : 100}),
        #    ("MRK-2", "adam", {"lr" : 1e-4}, "steplr", {"gamma" : 0.9, "step_size" : 4}),
        #    ("MRK-3", "adam", {"lr" : 1e-4, "amsgrad" : True}, "steplr", {"gamma" : 0.9}),
        #("MRK-4", "sgd" , {"lr" : 1e-4, "momentum" : 0.01, "nesterov" : True}, "steplr", {"gamma" : 0.9, "step_size" : 10}),
#        ("MRK-6", "sgd" , {"lr" : 1e-1, "momentum" : 0.4}, "steplr", {"gamma" : 0.9, "step_size" : 100}),
#         ("MRK-7", "sgd" , {"lr" : 1e-4, "momentum" : 0.03, "nesterov" : True}, "steplr", {"gamma" : 0.9, "step_size" : 100})
]

trains = []
optims = []
ana = Analysis()

n = 1
p = 0
for k in params:
    m1 = Grift()
    m1.o_edge  = {
            "top_edge" : "CrossEntropyLoss", #::(mean -> true)",
            "res_edge" : "CrossEntropyLoss", #::(mean -> true)"
    }
    m1.o_graph = {
            "ntops"  : "CrossEntropyLoss", #::(mean -> true)",
            "signal" : "CrossEntropyLoss", #::(mean -> true)"
    }
    m1.i_node  = ["pt", "eta", "phi", "energy"]
    m1.i_graph = ["met", "phi"]
    m1.device  = "cuda:" + str(p%n)

    opti = OptimizerConfig()
    opti.Optimizer = k[1]
    for t in k[2]: setattr(opti, t, k[2][t])
    opti.Scheduler = k[3]
    for t in k[4]: setattr(opti, t, k[4][t])
    p+=1
    trains.append(m1)
    optims.append(opti)

ana.AddSamples(root1, "tmp")
ana.AddEvent(x, "tmp")
ana.AddGraph(tt, "tmp")

for i in range(len(optims)): ana.AddModel(trains[i], optims[i], params[0][0] + "-"+str(i))

#ana.GraphCache = "./ProjectName/"
ana.Threads = 12
ana.kFolds = 10
ana.Epochs = 100
ana.TrainSize = 100
ana.TrainingDataset = "./ProjectName/sample.h5"
ana.kFold = [1] 
ana.BatchSize = 10
ana.DebugMode = True
#ana.BuildCache = True
ana.nBins = 400
ana.MaxRange = 400
ana.TrainSize = 90
ana.Targets = ["top_edge", "res_edge"]
#ana.SumOfWeightsTreeName = "sumWeights"
#ana.FetchMeta = True
ana.ContinueTraining = False
ana.Training = True
ana.Validation = False
ana.Evaluation = False
ana.SetLogY = False
ana.Start()

