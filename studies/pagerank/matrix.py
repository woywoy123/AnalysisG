from samples.training import Splitting
from samples.main import Samples
from training.config import MatrixCfg
import os

#smpl = Samples("/home/tnom6927/Downloads/mc16-full/")
#splt = Splitting("/home/tnom6927/Downloads/mc16-full/")

outdir  = "./"
indir = "./data/samples/"
cachedir = "./ProjectName/"
vars_ = ["pt", "eta", "phi", "energy"]
dset = "DataSets/graphdetectorlep.h5"

cfg = MatrixCfg("ProjectName", outdir, indir, cachedir)
#cfg.event("bsm4tops")
cfg.graph({"name" : "graphdetectorlep", "path" : "Graphs/", "build" : False})
cfg.base({"debug" : False, "input" : "training", "threads": 4, "intra" : 4})
cfg.plotting({"nbins" : 200, "range" : 400, "variables" : vars_, "target" : ["top_edge"], "logy" : False})
cfg.modes({"continue": True, "training" : True, "validation" : True, "evaluation": False})

param = [
    ["MRK1", "adam", {"lr" : 1e-4                                      }, "steplr", {"gamma" : 0.9, "step_size" : 100}], 
    ["MRK2", "adam", {"lr" : 1e-4, "amsgrad" : True                    }, "steplr", {"gamma" : 0.9, "step_size" : 100}], 
    ["MRK3", "sgd" , {"lr" : 1e-4, "momentum" : 0.01, "nesterov" : True}, "steplr", {"gamma" : 0.9, "step_size" : 100}],
    ["MRK4", "sgd" , {"lr" : 1e-2, "momentum" : 0.4                    }, "steplr", {"gamma" : 0.9, "step_size" : 100}]
]
epochs = 100
batch  = 40
kfolds = 10

kt = 0
for l in param:
    name, optm, parm, scd, prm = l
    for i in range(kfolds):
        cfg.training({"kfold" : [i+1], "kfolds" : kfolds, "splits" : 100, "epochs" : epochs, "batches": batch, "dataset" : dset})
        mdl = cfg.add_model(name)
        mdl.base("grift", "cuda")

        mdl.optimizer(optm, parm)
        mdl.scheduler(scd, prm)
        
        mdl.graph("met")
        mdl.graph("phi")
        mdl.graph("signal", "CrossEntropyLoss")
        mdl.graph("ntops", "CrossEntropyLoss")
        
        mdl.edge("top_edge", "CrossEntropyLoss", {"sum" : True, "smoothing" : 0.1})
        mdl.edge("res_edge", "CrossEntropyLoss", {"sum" : True, "smoothing" : 0.1})
        
        mdl.node("pt")
        mdl.node("eta")
        mdl.node("phi")
        mdl.node("energy")
        mdl.node("is_lep")
        mdl.node("is_b")
        mdl.dump()
        
        cfg.dump(kt)
        kt += 1 
    




