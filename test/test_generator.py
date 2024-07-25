from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.events.ssml_mc20.event_ssml_mc20 import SSML_MC20
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren, GraphTruthJets, GraphTruthJetsNoNu, GraphJets, GraphJetsNoNu, GraphDetectorLep, GraphDetector
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"

#x = BSM4Tops()
#tt = GraphChildren()
#tt = GraphTruthJets()
#tt = GraphJets()
#tt = GraphDetectorLep()
#
#m = RecursiveGraphNeuralNetwork()
#m.o_edge  = {
#    #    "res_edge" : "CrossEntropyLoss",
#        "top_edge" : "CrossEntropyLoss"
#}
#m.i_node  = ["pt", "eta", "phi", "energy", "is_lep", "is_b"]
#m.i_graph = ["met", "phi"]
#m.device  = "cuda:0"
#
#op = OptimizerConfig()
#op.Optimizer = "adam"
#op.lr = 1e-6
#
#ana = Analysis()
#ana.TrainingDataset = "./ProjectName/dataset"
#ana.AddSamples(root1, "tmp")
#ana.AddEvent(x, "tmp")
#ana.AddGraph(tt, "tmp")
#ana.AddModel(m, op, "test")
#ana.kFolds = 10
#ana.kFold = [1]
#ana.Targets = ["top_edge"]
#ana.ContinueTraining = False
#ana.Evaluation = False
#ana.Validation = False
#ana.DebugMode = False
#ana.Epochs = 1000
#ana.MaxRange = 400
#ana.TrainSize = 95
#ana.Start()

from AnalysisG.core.io import IO

root1 = "/home/tnom6927/Downloads/mc20/*"

t = SSML_MC20()
#x = IO([root1])
#x.Trees = ["reco"]
#x.Leaves = ["el_charge"]
#for i in x:
#    print(i)

ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(t, "tmp")
ana.Start()


