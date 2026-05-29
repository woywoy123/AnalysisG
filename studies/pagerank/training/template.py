from training.atomics import *
from training.methods import *
from training.config import *

from AnalysisG import Analysis
from AnalysisG.core.lossfx import OptimizerConfig

class TrainingCfg(DefaultCfg):

    def __init__(self, model_name = ""):
        DefaultCfg.__init__(self, [model_method])

        self.name             = model_name
        self.kFold            = None
        self.kFolds           = None
        self.Epochs           = None

        self.BatchSize        = None
        self.TrainSize        = None
        self.TrainingDataset  = None

        self.Training         = None
        self.Validation       = None
        self.Evaluation       = None
        self.ContinueTraining = None

class SchedulerCfg(DefaultCfg):

    def __init__(self, scheduler_name):
        DefaultCfg.__init__(self, [scheduler_method])
        self.name       = scheduler_name
        self.step_size  = None
        self.step_gamma = None
        self.check()
    
    def __compile__(self, op):
        op.Scheduler = self.name
        SetEnvironment(op, self, "step_size")
        SetEnvironment(op, self, "step_gamma")


class OptimizerCfg(DefaultCfg):

    def __init__(self, optimizer_name):
        DefaultCfg.__init__(self)

        self.name         = optimizer_name
        self.lr           = None
        self.lr_decay     = None

        self.alpha        = None
        self.beta         = None
        self.eps          = None
        self.weight_decay = None

        self.momentum  = None
        self.dampening = None
        self.nesterov  = None
        self.centered  = None

        self.amsgrad  = None
        self.max_iter = None
        self.max_eval = None

        self.history_size = None
        self.tolerance_grad = None
        self.tolerance_change = None
        self.initial_accumulator_value = None

    def __compile__(self):
        self.optim = OptimizerConfig()
        self.optim.Optimizer = optimizer_method(self.name)
        x = [
                "lr", "lr_decay", "beta", "eps", "weight_decay", "momentum", "dampening", "nesterov",
                "centered", "amsgrad", "max_iter", "max_eval", "history_size", "tolerance_grad",
                "tolerance_change", "initial_accumulator_value", "alpha"
        ]
        for i in x: SetEnvironment(self.optim, self, i)

class FeatureCfg:

    def __init__(self, fname, loss = None):
        self.name = fname
        self.loss = loss

    def __str__(self): return str(self.loss)

class LossCfg(DefaultCfg):

    def __init__(self, name, flag = None):
        DefaultCfg.__init__(self)
        self.name = name
        self.flag = flag
        self.config = loss_method(self.name)

        if self.flag is None: return 
        name = self.config + "::("
        for i in self.flag: 
            name += loss_string_method(i, self.flag[i]) + " -> " + str(self.flag[i])
            if len(self.flag) > 1: name += " | "
        self.config = name.rstrip(" | ") + ")" 

    def __str__(self): return self.config


class ModelCfg(DefaultCfg):

    def __init__(self, model_name, optim_name = None, sched_name = None):
        DefaultCfg.__init__(self)
        self.model  = model_method(model_name, True)
        self.device = None

        self.edges = {"in" : {}, "out" : {}}
        self.nodes = {"in" : {}, "out" : {}}
        self.graph = {"in" : {}, "out" : {}}

        self.optimizer = OptimizerCfg(optim_name) if optim_name is not None else None
        self.scheduler = SchedulerCfg(sched_name) if sched_name is not None else None
   
    def ConfigLoss(self, name, flags = None): return LossCfg(name, flags)
    
    def EdgeFeature(self, name, loss = None):
        self.edges["in" if loss is None else "out"][name] = FeatureCfg(name, loss)
        
    def NodeFeature(self, name, loss = None):
        self.nodes["in" if loss is None else "out"][name] = FeatureCfg(name, loss)
       
    def GraphFeature(self, name, loss = None):
        self.graph["in" if loss is None else "out"][name] = FeatureCfg(name, loss)
  
    def __compile__(self):
        SetEnvironment(self.model, self, "device")
        if len(self.edges["in"]): self.model.i_edge  = list(self.edges["in"])
        if len(self.nodes["in"]): self.model.i_node  = list(self.nodes["in"])
        if len(self.graph["in"]): self.model.i_graph = list(self.graph["in"])

        if len(self.edges["out"]): self.model.o_edge  = {i : str(self.edges["out"][i]) for i in self.edges["out"]}
        if len(self.nodes["out"]): self.model.o_node  = {i : str(self.nodes["out"][i]) for i in self.nodes["out"]}
        if len(self.graph["out"]): self.model.o_graph = {i : str(self.graph["out"][i]) for i in self.graph["out"]}

        if self.optimizer is not None: self.optimizer.__compile__()
        if self.scheduler is not None: self.scheduler.__compile__(self.optimizer.optim)
        
