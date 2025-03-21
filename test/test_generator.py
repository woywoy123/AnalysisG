from AnalysisG import Analysis
from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.events.ssml_mc20.event_ssml_mc20 import SSML_MC20
from AnalysisG.graphs.bsm_4tops import GraphTops, GraphChildren, GraphTruthJets, GraphTruthJetsNoNu, GraphJets, GraphJetsNoNu, GraphDetectorLep, GraphDetector
from AnalysisG.models import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/sample/*"

x = BSM4Tops()
#tt = GraphChildren()
#tt = GraphTruthJets()
tt = GraphJets()
tt.PreSelection = True
#tt = GraphDetector()

#m = RecursiveGraphNeuralNetwork()
m = Grift()

m.o_edge  = {"res_edge" : "CrossEntropyLoss", "top_edge" : "CrossEntropyLoss"}
m.o_graph = {"signal"   : "CrossEntropyLoss", "ntops"    : "CrossEntropyLoss"}

m.i_node  = ["pt", "eta", "phi", "energy", "is_lep", "is_b"]
m.i_graph = ["met", "phi"]
m.device  = "cuda"

op = OptimizerConfig()
op.Optimizer = "adam"
op.lr = 1e-3

ana = Analysis()
ana.PreTagEvents = True

#ana.FetchMeta = True
ana.TrainingDataset = "./ProjectName/dataset"
ana.AddSamples(root1, "tmp")
ana.AddEvent(x,  "tmp")
ana.AddGraph(tt, "tmp")
ana.AddModel(m, op, "test")
ana.kFolds = 10
ana.kFold = [1]
ana.Targets = ["top_edge"]
ana.ContinueTraining = False
ana.GraphCache = "./ProjectName/GraphCache/"
ana.Evaluation = True
ana.Validation = True
ana.DebugMode = False
ana.Epochs = 1000
ana.MaxRange = 400
ana.TrainSize = 80
ana.Threads = 12
ana.Start()

