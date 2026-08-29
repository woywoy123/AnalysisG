import AnalysisG
from AnalysisG.core import Analysis
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events.bsm_4tops import *
from AnalysisG.core.lossfx import *
from AnalysisG.metrics import *
from AnalysisG.models import *

ev = BSM4Tops()
gr = GraphTruthJets()

root1 = "./samples/dilepton/*"
root2 =  "/home/tnom6927/scratch/data/*"

gn = Grift()
gn.name = "MRK-1"
gn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
gn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
gn.i_node  = ["pt", "eta", "phi", "energy", "charge"]
gn.i_graph = ["met", "phi"]
gn.device = "cuda:0"

base_dir = "/home/tnom6927/scratch/"

mx = AccuracyMetric()
mx.RunNames = {
        "MRK-1::epoch-1::k-1" : base_dir + "MRK-1/state/epoch-1/kfold-1_model.pt", 
        "MRK-1::epoch-2::k-1" : base_dir + "MRK-1/state/epoch-2/kfold-1_model.pt", 
        "MRK-1::epoch-3::k-1" : base_dir + "MRK-1/state/epoch-3/kfold-1_model.pt", 
}

mx.Variables = [
        "MRK-1::data::node::pt",
        "MRK-1::data::node::eta",
        "MRK-1::data::node::phi",
        "MRK-1::data::node::energy",

        "MRK-1::truth::graph::ntops", 
        "MRK-1::prediction::extra::ntops_score", 

        "MRK-1::truth::edge::top_edge",
        "MRK-1::prediction::extra::top_edge_score", 

        "MRK-1::data::node::index",
        "MRK-1::data::edge::index", 
]

#my = PageRankMetric()
#my.RunNames = {
#        "MRK-1::epoch-1::k-10" : base_dir + "MRK-1/state/epoch-1/kfold-10_model.pt", 
#        "MRK-1::epoch-2::k-10" : base_dir + "MRK-1/state/epoch-2/kfold-10_model.pt", 
#        "MRK-1::epoch-3::k-10" : base_dir + "MRK-1/state/epoch-3/kfold-10_model.pt", 
#}
#
#my.Variables = [
#        "MRK-1::data::node::pt",
#        "MRK-1::data::node::eta",
#        "MRK-1::data::node::phi",
#        "MRK-1::data::node::energy",
#
#        "MRK-1::truth::graph::ntops", 
#        "MRK-1::prediction::extra::ntops_score", 
#
#        "MRK-1::truth::edge::top_edge",
#        "MRK-1::prediction::extra::top_edge_score", 
#
#        "MRK-1::data::node::index",
#        "MRK-1::data::edge::index", 
#]
#
ana = Analysis()
ana.AddEvent(ev, "tmp")
ana.AddGraph(gr, "tmp")
ana.AddSamples(root2, "tmp")
ana.TrainSize = 50
#ana.BatchSize = 2
##ana.kFold = [1, 2, 3]
ana.kFolds = 10
ana.BuildCache = True
ana.GraphCache = "./ProjectName/"
ana.TrainingDataset = "./ProjectName/sample.h5"
ana.Threads = 2

ana.AddMetric(mx, gn)
#ana.AddMetric(my, gn)

ana.Validation = True
ana.Evaluation = True
ana.Training   = True
ana.Start()
##ana.DebugMode = True

#mx.InterpretROOT("./ProjectName/metrics/pagerank/epoch-1/Grift-MRK-1/kfold-1.root")

#ana.AddModel(gn, opti, "MRK-1")


