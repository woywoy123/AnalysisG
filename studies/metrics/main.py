import AnalysisG
from AnalysisG import Analysis
from AnalysisG.core.tools import Tools
from AnalysisG.metrics import AccuracyMetric
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.models import *

from atomics import *


base_dir   = "/scratch/tnom6927/"
base_model = "Grift"

train_path = base_dir + "model-package/gnn-update"
graph_index_tr = base_dir + "model-package/Graphs/graph_jets_detector_lep"
graph_index_ev = base_dir + "model-package/evaluation/graph_jets_detector_lep"
graph_train    = base_dir + "model-package/Graphs/GraphDetectorLep_train.h5"

runs = []
tl = Tools()
for i in tl.ls(train_path, ""):
    if not i.endswith(".pt"): continue
    if "optimizer" in i: continue
    ss = Sessions(train_path)
    ss.parse(i)
    runs.append(ss)

varx = [
    ["truth", "graph",    "ntops"],
    ["truth", "edge" , "top_edge"],
    ["prediction", "extra", "ntops_score"   ],
    ["prediction", "extra", "top_edge_score"],
    ["data", "node", "index"],
    ["data", "edge", "index"], 
    ["data", "graph", "index"]
]

prm = ModelParams()
prm.variables = varx
prm.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
prm.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
prm.i_node  = ["pt", "eta", "phi", "energy"]
prm.i_graph = ["met", "phi"]
prm.train_set = graph_train
prm.graph_trn = graph_index_tr
prm.graph_evl = graph_index_ev
prm.batch_size  = 50

ml = ModelEnv(runs, prm)
ml.compile()

rn = {i.tag : i.abs for i in ml.sessions}
vr = list(set(sum([i.variables for i in ml.sessions], [])))
for i in range(100):

    ana = Analysis()
    ana.Threads   = 2
    ana.BatchSize = prm.batch_size
    ana.GraphCache = prm.graph_trn
    ana.TrainingDataset = prm.train_set

    ana.Validation = True
    ana.Training   = True

    gn1 = Grift()
    gn1.name    = "Grift-MRK-2"
    gn1.o_edge  = prm.o_edge
    gn1.o_graph = prm.o_graph
    gn1.i_node  = prm.i_node
    gn1.i_graph = prm.i_graph
    gn1.device = "cuda:0"

    _rn = {k : rn[k] for k in rn if "epoch-" + str(i+1) + "::" in k and gn1.name in k}
    _rv = [k         for k in vr if gn1.name in k]

    mx1 = AccuracyMetric()
    mx1.RunNames  = _rn
    mx1.Variables = _rv
    ana.AddMetric(mx1, gn1)

    #ana.GraphCacheSplit = {"evaluation" : prm.graph_evl}
    ana.Evaluation = False
    ana.Start()



#gn2 = Grift()
#gn2.name    = "Grift-MRK-2"
#gn2.o_edge  = prm.o_edge
#gn2.o_graph = prm.o_graph
#gn2.i_node  = prm.i_node
#gn2.i_graph = prm.i_graph
#gn2.device   = "cuda:1"
#
#mx2 = AccuracyMetric()
#mx2.RunNames = {i.tag : i.abs for i in ml.sessions if gn2.name in i.tag}
#mx2.Variables = [i for i in vr if gn2.name in i]
#ana.AddMetric(mx2, gn2)


