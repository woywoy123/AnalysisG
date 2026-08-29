import AnalysisG
from AnalysisG import *
from AnalysisG.core import Analysis
from AnalysisG.models  import Grift
from AnalysisG.metrics import AccuracyMetric

from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events.bsm_4tops import *

from atomics import *

base_dir   = "/CERN/thesis-data/gnn-model/"
base_model = "Grift"

rn = Runtime(base_dir)
rn.training.path   = "training/base/GraphDetectorLep/"
rn.training.graphs("graph_jets_detector_lep", GraphDetectorLep())
rn.meta.meta  = "data/meta"

rn.evaluation.path = "data/evaluation"
rn.evaluation.graphs("data/evaluation", GraphDetectorLep())
rn.kfolds.path = "data/k-folds/GraphdetectorLep_train.h5"
rn.kfolds.folds = 10
rn.kfolds.epoch = 200
params = [
    ("MRK-1", "adam"   , {"lr" : 1e-3}                                         , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-2", "adam"   , {"lr" : 1e-3, "amsgrad" : True}                       , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-3", "sgd"    , {"lr" : 1e-3}                                         , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-4", "sgd"    , {"lr" : 1e-3, "momentum": 0.20 , "nesterov" : True}   , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-5", "sgd"    , {"lr" : 1e-3, "momentum": 0.10 , "nesterov" : True}   , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-6", "adamw"  , {"lr" : 1e-3, "amsgrad" : True}                       , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-7", "rmsprop", {"lr" : 1e-3, "momentum": 0.10 , "centered": True}    , "steplr", {"gamma" : 0.85, "step_size"  : 10}), 
    ("MRK-8", "adagrad", {"lr" : 1e-3, "amsgrad": True}                        , "steplr", {"gamma" : 0.85, "step_size"  : 10})
]


varx = [
    ["truth", "graph",    "ntops"],
    ["truth", "edge" , "top_edge"],
    ["prediction", "extra", "ntops_score"   ],
    ["prediction", "extra", "top_edge_score"],
    ["data", "node", "index"],
    ["data", "edge", "index"], 
    ["data", "graph", "index"]
]

for i in params: 
    cfg = Model(i[0], Grift).optimizer(i[1], i[2]).scheduler(i[3], i[4]).device(0)
    cfg = rn.models.add(cfg)
    cfg.variables = varx
    cfg.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
    cfg.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
    cfg.i_node  = ["pt", "eta", "phi", "energy"]
    cfg.i_graph = ["met", "phi"]
    cfg.batch_size  = 50
    break






print(rn.compile())







exit()

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


