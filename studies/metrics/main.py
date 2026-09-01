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
rn.training.path  = "training/base/GraphDetectorLep/"
rn.training.graphs("graph_jets_detector_lep", GraphDetectorLep())
rn.meta.meta  = "data/meta"

rn.evaluation.path = "data/evaluation"
rn.evaluation.graphs("data/evaluation", GraphDetectorLep())

rn.kfolds.path = "data/k-folds/GraphdetectorLep_train.h5"
rn.kfolds.split = {"training" : , "validation", "evaluation"]

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


pred_extra = ["top_edge_score", "ntops_score"]
data_node  = ["pt", "eta", "phi", "energy", "is_lep", "is_b", "index"]
data_edge  = ["index"]
data_graph = ["index"]

batch_node  = ["index"]
batch_graph = ["index"]

truth_graph = ["ntops"]
truth_edge  = ["top_edge"]

for i in params: 
    cfg = Model(i[0], Grift).optimizer(i[1], i[2]).scheduler(i[3], i[4]).device(0)
    cfg = rn.models.add(cfg)

    for k in data_node:  cfg.node( "data", k)
    for k in data_edge:  cfg.edge( "data", k)
    for k in data_graph: cfg.graph("data", k)
    for k in pred_extra: cfg.extra("prediction", k)

    for k in batch_node:  cfg.node( "batch", k)
    for k in batch_graph: cfg.graph("batch", k)

    for k in truth_edge:  cfg.edge( "truth", k)
    for k in truth_graph: cfg.graph("truth", k)

    cfg.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
    cfg.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
    cfg.batch_size  = 50
    break



