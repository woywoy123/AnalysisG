#import torch
import AnalysisG
from AnalysisG import Analysis
from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events import SSML_MC20
from AnalysisG.models import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/mc20/user.bdong.503575.MGPy8EG.DAOD_PHYS.e8307_s3797_r13144_p6266.tom_sample_v01_output/user.bdong.41003314._000001.output.root"
root1 = "/home/tnom6927/Downloads/mc16/ttbar/user.tnommens.40945479._000001.output.root"
#root1 = "/home/tnom6927/Downloads/mc16/ttH_tttt_m400/*"

x = BSM4Tops()
#x = SSML_MC20()
#tt = GraphChildren()
#tt = GraphTruthJets()
#tt = GraphTruthJets()
#tt = GraphJets()

#tt = GraphDetectorLep() 

tt = GraphDetector()
tt.PreSelection = False
tt.ForceMatch = True
tt.NumCuda = 1

params = [
    #    ("MRK-1", "adam", {"lr" : 1e-4}, "steplr", {"gamma" : 0.9, "step_size" : 1}),
    #("MRK-2", "adam", {"lr" : 1e-6}, "steplr", {"gamma" : 0.9}),
    #("MRK-3", "adam", {"lr" : 1e-6, "amsgrad" : True}, "steplr", {"gamma" : 0.9}),
    #("MRK-4", "sgd" , {"lr" : 1e-3}),
    #("MRK-5", "sgd", {"lr" : 1e-6}),
    ("MRK-6", "sgd", {"lr" : 1e-4, "momentum" : 0.1}, "steplr", {"gamma" : 0.9}),
    #    ("MRK-7", "sgd", {"lr" : 1e-6, "momentum" : 0.01, "dampening" : 0.01})
]

trains = []
optims = []
ana = Analysis()

n = 1
p = 0
for k in params:
    m1 = Grift()
#    m1.PageRank = True
    m1.o_edge  = {
            "top_edge" : "CrossEntropyLoss",
            "res_edge" : "CrossEntropyLoss"
    }
    m1.o_graph = {
            "ntops"  : "CrossEntropyLoss",
            "signal" : "CrossEntropyLoss"
    }
    m1.i_node  = ["pt", "eta", "phi", "energy", "charge"]
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
ana.Threads = 2
#ana.AddSamples(ttZ, "ttZ")
ana.AddEvent(x, "tmp")
ana.AddGraph(tt, "tmp")

#ana.AddEvent(x, "ttZ")
#ana.AddGraph(tt, "ttZ")

for i in range(len(optims)): ana.AddModel(trains[i], optims[i], params[0][0] + "-"+str(i))

ana.kFolds = 10
ana.Epochs = 100
ana.TrainingDataset = "./ProjectName/sample.h5"
ana.Targets = ["top_edge", "res_edge"]
ana.MaxRange = 400
ana.GraphCache = "./ProjectName/"
ana.kFold = [1] #, 2, 3, 4] #6, 7, 3, 5, 9]
ana.TrainSize = 80
ana.BatchSize = 2
ana.DebugMode = True
ana.nBins = 400
ana.ContinueTraining = True
#ana.SumOfWeightsTreeName = "sumWeights"
#ana.FetchMeta = True
ana.DebugMode  = False
ana.Threads = 1
ana.Validation = False
ana.Evaluation = False
ana.Start()

