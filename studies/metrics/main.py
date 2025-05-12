import AnalysisG
from AnalysisG.metrics import AccuracyMetric
from AnalysisG.models import *

def default(tl):
    tl.Style = "ATLAS"
    tl.DPI = 300
    tl.TitleSize = 15
    tl.AutoScaling = True
    tl.LegendSize = 10
    tl.yScaling = 5 
    tl.xScaling = 10
    tl.FontSize = 10
    tl.AxisSize = 10
    tl.LineWidth = 1

def make_metrics(graph_cache, train_set, batch_size, threads, RunNames, variables):
    mx = AccuracyMetric()
    mx.RunNames = RunNames
    mx.Variables = variables

    gn = Grift()
    gn.name = "Grift-MRK-1"
    gn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
    gn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
    gn.i_node  = ["pt", "eta", "phi", "energy"]
    gn.i_graph = ["met", "phi"]
    gn.device = "cuda:0"

    ana = Analysis()
    ana.Threads = threads
    ana.BatchSize = batch_size
    ana.AddMetric(mx, gn)
    ana.GraphCache = graph_cache
    ana.TrainingDataset = train_set
    ana.Validation = True
    ana.Evaluation = True
    ana.Training   = True
    ana.Start()




base_name = "Grift"
graph_cache = "./ProjectName/"
base_dir = "./ProjectName/Grift/"
train_set = "./ProjectName/sample.h5"
batch_size = 10
threads = 2
epochs = 10
kfold = 10

RunNames = {}
Variables = []
for mrk in ["MRK-1", "MRK-2", "MRK-3", "MRK-4", "MRK-5"]:
    name = base_name + "-" + mrk
    Variables += [
            name + "::truth::graph::ntops", 
            name + "::prediction::extra::ntops_score", 
    
            name + "::truth::edge::top_edge",
            name + "::prediction::extra::top_edge_score", 
    
            name + "::data::node::index",
            name + "::data::edge::index", 
    ]

    for ep in range(epochs):
        for kf in range(kfold):
            key = base_name + "-" + mrk + "::epoch-" + str(ep+1) + "::k-" + str(kf+1)
            pth = base_dir + mrk + "/epoch-" + str(ep+1) + "/kfold-" + str(kf+1) + "_model.pt"

#make_metrics(graph_cache, train_set, batch_size, threads, RunNames, Variables)



sl = AccuracyMetric()
sl.default_plt = default
sl.InterpretROOT("./accuracy/", epochs = [], kfolds = [3])

