from AnalysisG import Analysis
from AnalysisG.graphs.ssml_mc20 import GraphJets, GraphDetector
#from AnalysisG.graphs.bsm_4tops import GraphJets
from AnalysisG.events import SSML_MC20

from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.models import Grift

ttbar = "/home/tnom6927/Downloads/mc20/user.rqian.503575.MGPy8EG.DAOD_PHYS.e8307_s3797_r13167_p6490.2024-11-23_output"

ev = SSML_MC20()
gr = GraphDetector()
md = Grift() #RecursiveGraphNeuralNetwork()
md.o_edge = {
        "top_edge" : "CrossEntropyLoss",
        "res_edge" : "CrossEntropyLoss"
}

md.o_graph = {
        "ntops"  : "CrossEntropyLoss",
        "signal" : "CrossEntropyLoss"
}

#md.i_node  = ["pt", "eta", "phi", "energy"]
#md.i_graph = ["met", "phi"]
md.device  = "cuda:0"
md.checkpoint_path = "./ProjectName/Grift/MRK-1-0/state/epoch-1/kfold-1_model.pt"

#k = ("MRK-1", "adam", {"lr" : 1e-4}),
#opti = OptimizerConfig()
#opti.Optimizer = k[0][1]
#for t in k[0][2]: setattr(opti, t, k[0][2][t])

ana = Analysis()
#ana.kFold = [1]
#ana.kFolds = 10
#ana.TrainingDataset = "./ProjectName/sample.h5"

#ana.AddSamples(bsm  , "bsm")
#ana.AddSamples(tttt , "tttt")
ana.AddSamples(ttbar, "ttbar")
#ana.AddEvent(ev, "bsm")
#ana.AddEvent(ev, "tttt")
ana.AddEvent(ev, "ttbar")

#ana.AddGraph(gr, "bsm")
#ana.AddGraph(gr, "tttt")
ana.AddGraph(gr, "ttbar")

#ana.Threads = 1
#ana.Epochs = 100
#ana.TrainSize = 80
ana.BatchSize = 10
#ana.AddModel(md, opti, "run-name")
ana.AddModelInference(md, "run-name")
ana.GraphCache = "./ProjectName/"
#ana.Targets = ["top_edge"]
#ana.Validation = False
#ana.Evaluation = False
ana.Start()
