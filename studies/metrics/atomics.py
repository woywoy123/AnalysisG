from AnalysisG.core import IO
from AnalysisG.core import Tools
from AnalysisG.core import Notification
from AnalysisG.core.lossfx import OptimizerConfig

from custom_models import ModelParams, session, KFolds
from samples import *
import pickle

class meta:

    def __init__(self, base_dir = None):
        self.cache       = base_dir 
        self.training    = "training.txt"
        self.evaluation  = "evaluation.txt"
        self.fullsamples = "full-log.txt"

    @property
    def meta(self): return self.cache
    @meta.setter
    def meta(self, u): self.cache = pfix(self.cache, u)

class Samples:

    def __init__(self, type_, base_dir = None):
        self.base_dir = base_dir
        self.type     = type_
        self.path     = None
        self.data     = {}
        self._meta    = None
    
    def graphs(self, path, impl):
        self.impl  = impl
        self.graph = path

    @property
    def _fdir(self): return pfix(self.base_dir, self.path)

class Model:
    def __init__(self, name, impl = None):
        self.name = name
        self.impl = impl
        self.optim = OptimizerConfig()
        self._device = None
        self._kgen  = None
        self._data  = {}
        self._kfold = {}

    def add(self, model):
        self._data[model.name] = ModelParams(model, self)
        self._kfold[model.name] = {
            k : KFolds(self._kgen.split[k], self._kgen).model(model) for k  in self._kgen.split
        }
        return self._data[model.name]

    def optimizer(self, name, params):
        self.optim.Optimizer = name
        for i in params: setattr(self.optim, i, params[i])
        return self
    
    def scheduler(self, name, params):
        self.optim.Scheduler = name
        for i in params: setattr(self.optim, i, params[i])
        return self

    def device(self, cu_dvi = 0):
        self._device = "cuda:" + str(cu_dvi)
        return self

class Runtime:

    def __init__(self, base_dir = None):
        self.notf       = Notification("Runtime")
        self.evaluation = Samples("evaluation", base_dir)
        self.training   = Samples("training"  , base_dir)
        self.metrics    = Samples("metrics"   , base_dir)

        self.kfolds     = KFolds(base_dir)
        self.models     = Model("model", base_dir)
        self.models._kgen = self.kfolds
        self.meta       = meta(base_dir)
        self.stats      = Statistics(base_dir)
        try: self.stats = pickle.load(open("cache.pkl", "rb"))
        except FileNotFoundError: pass

    def SampleParams(self):
        camp = -1
        filepath = self.meta.meta + "/" + self.meta.fullsamples
        f = open(filepath, "r")
        lines = [i.strip() for i in f.readlines() if len(i.strip()) > 1]

        for i in lines:
            dc = {}
            if "mc16" in i and "/" not in i: camp = [k for k in i.split(":") if "mc16" in k][0].replace(" ", "").replace("-", "")
            if camp not in self.stats.processes: self.stats.processes[camp] = {}
            for t in i.split(", "): 
                k, v = t.rsplit(": ")
                dc[k.rstrip()] = v.rstrip().replace(" ", "")
            if "process" not in dc: continue 
            logical, dname, prc = clk(dc["directory"]), sani(dc["DatasetName"]), dc["process"]
            if prc not in self.stats.processes[camp]: self.stats.processes[camp][prc] = Process(prc)
            self.stats.processes[camp][prc].add(dc)
            self.stats.daods[logical] = self.stats.processes[camp][prc].datasets[dname].daods[logical]

        for i in open(self.meta.meta + "/" + self.meta.training, "r").readlines():
            if "user." not in i: continue
            logical = clk(sani(i))
            logical = self.stats.daods[logical]
            logical.train = True
            self.stats.train.append(logical)
        
        for i in open(self.meta.meta + "/" + self.meta.evaluation, "r").readlines():
            if "user." not in i: continue
            logical = clk(sani(i))
            logical = self.stats.daods[logical]
            logical.eval = True
            self.stats.eval.append(logical)
   
    def ModelTasks(self):
        ct = Tools() 
    
        x = 0
        for i in self.models._data.values():
            fd  = self.training._fdir + "/" + i.model.impl.__name__ + "/" + i.model.name +  "/" 
            for kt in self.models._kfold:
                for t in self.models._kfold[kt]:
                    for l in self.models._kfold[kt][t].combinations():

                        pth = fd + "state/" + l._kdir
                        fk = l._ddir + "/" + t + ".root"

                        try: self.stats.kfolds[fk]; continue
                        except: pass 

                        fwt, fkt = ct.is_file(pth), ct.is_file(fk)
                        l.mode = t
                        
                        self.stats.kfolds[fk] = l
                        try:    self.stats.models[l.modeln][l.epoch][l.folds] = l
                        except: self.stats.models[l.modeln][l.epoch] = {l.folds : l}

                        if not fwt: continue
                        if not fkt: continue
                        x += 1
                        if x % 50 == 0: pickle.dump(self.stats, open("cache.pkl", "wb"))

                        io = IO()
                        io.Files = [fk]
                        io.Trees = "accuracy_" + t
                        io.Leaves = "dsid"
                        l.ndata = len([f for f in io])
                        if not l.ndata: print("BROKEN"); continue

                        l.missing = False
                        l.smpl_l  = fk

        pickle.dump(self.stats, open("cache.pkl", "wb"))
        return self.stats

