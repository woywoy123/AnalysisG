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
#graph_index = base_dir + "model-package/Graphs/graph_jets_detector_lep"
graph_index = base_dir + "model-package/evaluation/graph_jets_detector_lep"
graph_train = base_dir + "model-package/Graphs/GraphDetectorLep_train.h5"

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
    ["data", "edge", "index"]
]

prm = ModelParams()
prm.variables = varx
prm.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
prm.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
prm.i_node  = ["pt", "eta", "phi", "energy"]
prm.i_graph = ["met", "phi"]
prm.train_set = graph_train
prm.graph_set = graph_index
prm.batch_size  = 10

ml = ModelEnv(runs, prm)
ml.compile()

for i in ml.sessions:
    mx = AccuracyMetric()
    mx.RunNames  = {i.tag : i.abs}
    mx.Variables = i.variables

    gn = Grift()
    gn.name    = i.mdl + "-" + i.mrk
    gn.o_edge  = prm.o_edge
    gn.o_graph = prm.o_graph
    gn.i_node  = prm.i_node
    gn.i_graph = prm.i_graph
    gn.device = "cuda:0"

    ana = Analysis()
    ana.Threads   = 44
    ana.BatchSize = prm.batch_size
    ana.GraphCache = prm.graph_set
    ana.AddMetric(mx, gn)

    #ana.TrainingDataset = prm.graph_set
    ana.Validation = False
    ana.Training   = False
    ana.Evaluation = True
    ana.Start()

    exit()



