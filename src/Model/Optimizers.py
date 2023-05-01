import torch 
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from AnalysisG.Settings import Settings

class OptimizerWrapper(Settings):

    def __init__(self):
        self.Caller = "OPTIMIZER"
        Settings.__init__(self)
        self.train = False
        self._op = None 
        self._sc = None
        self._mod = None
 
    @property 
    def SetOptimizer(self):
        self._mod = self.Model._Model
        self._pth = self.OutputDirectory + "/" + self.RunName
        if self.Optimizer == "ADAM": self._op = torch.optim.Adam(self._mod.parameters(), **self.OptimizerParams)
        elif self.Optimizer == "SDG": self._op = torch.optim.SGD(self._mod.parameters(), **self.OptimizerParams)
        else: return False
        return True 

    @property
    def dump(self):
        dct = {}
        dct["epoch"] = self.Epoch
        dct["optim"] = self._op.state_dict()
        if self._sc is not None: dct["sched"] = self._sc.state_dict()
        torch.save(dct, self._pth + "/" + str(self.Epoch) + "/TrainingState.pth")

    @property
    def load(self):
        v = torch.load(self._pth + "/" + str(self.Epoch) + "/TrainingState.pth")
        self.Epoch = v["epoch"]
        self._op.load_state_dict(v["optim"])
        if self._sc is not None: self._sc.load_state_dict(v["sched"])

    @property
    def step(self): 
        if self.train: self._op.step()
        else: None
    
    @property
    def zero(self): 
        if self.train: self._op.zero_grad()
        else: None
    
    @property
    def SetScheduler(self):
        if self.Scheduler == "ExponentialLR": self._sc = ExponentialLR(**self.SchedulerParams)
        if self.Scheduler == "CyclicLR": self._sc = CyclicLR(**self.SchedulerParams)
        if self._sc == None: return False
        return True
    
    @property
    def stepsc(self): 
        if self._sc is None: return 
        self._sc.step()
