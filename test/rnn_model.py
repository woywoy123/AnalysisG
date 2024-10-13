import AnalysisG
from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.events import SSML_MC20
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.models import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/mc20/user.bdong.503575.MGPy8EG.DAOD_PHYS.e8307_s3797_r13144_p6266.tom_sample_v01_output/user.bdong.41003314._000001.output.root"
x = BSM4Tops()
#x = SSML_MC20()
#tt = GraphChildren()
#tt = GraphTruthJets()
#tt = GraphDetector()
tt = GraphJets()

params = [
    ("MRK-1", "adam", {"lr" : 1e-4}),
    #("MRK-2", "adam", {"lr" : 1e-6}),
    #    ("MRK-3", "adam", {"lr" : 1e-6, "amsgrad" : True}),
    #("MRK-4", "sgd" , {"lr" : 1e-3}),
    #("MRK-5", "sgd", {"lr" : 1e-6}),
    #    ("MRK-6", "sgd", {"lr" : 1e-4, "momentum" : 0.1}),
    #    ("MRK-7", "sgd", {"lr" : 1e-6, "momentum" : 0.01, "dampening" : 0.01})
]

trains = []
optims = []
ana = Analysis()
p = 0
for k in params:
#    m1 = RecursiveGraphNeuralNetwork()
    m1 = Grift()
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
    m1.device  = "cuda:" + str(p%1)

    opti = OptimizerConfig()
    opti.Optimizer = k[1]
    for t in k[2]: setattr(opti, t, k[2][t])
    p+=1
    trains.append(m1)
    optims.append(opti)

ana.AddSamples(root1, "tmp")
ana.AddEvent(x, "tmp")
ana.AddGraph(tt, "tmp")

for i in range(len(optims)): ana.AddModel(trains[i], optims[i], params[0][0] + "-"+str(i))

##ana.kFolds = 10
ana.Epochs = 100
ana.TrainingDataset = "./ProjectName/sample.h5"
#ana.Targets = ["top_edge", "res_edge"]
ana.GraphCache = "./ProjectName/"
ana.kFold = [6]
ana.MaxRange = 500
ana.TrainSize = 80
ana.BatchSize = 1
ana.ContinueTraining = False
#ana.SumOfWeightsTreeName = "sumWeights"
#ana.FetchMeta = True
ana.DebugMode  = False
ana.Validation = False
ana.Evaluation = False
ana.Start()

