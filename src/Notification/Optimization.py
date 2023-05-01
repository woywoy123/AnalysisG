from .Notification import Notification 
from AnalysisG.IO import UnpickleObject

class _Optimizer(Notification):
    
    def __init__(self):
        pass

    @property
    def _NoModel(self):
        if self.Model is not None: return False
        self.Warning("No Model was given.")
        return True

    @property
    def _NoSampleGraph(self):
        l = len(self)
        if l == 0: return self.Warning("No Sample Graphs found")
        self.Success("Found " + str(l) + " Sample Graphs") 
        self._nsamples = l
        return False

    @property
    def _notcompatible(self):
        return self.Failure("Model not compatible with given input graph sample.")

    @property 
    def _setoptimizer(self):
        if not self._op.SetOptimizer: return self.Failure("Invalid Optimizer.")
        return not self.Success("Set the optimizer to: " + self._op.Optimizer)

    @property 
    def _setscheduler(self):
        if not self._op.SetScheduler: return 
        self.Success("Set the scheduler to: " + self._op.Scheduler)

    @property 
    def _searchtraining(self):
        self.Epoch = 0
        pth = self.OutputDirectory + "/" + self.RunName
        f = self.ls(pth)
        
        if not self.ContinueTraining: return 
        if len(f) == 0: return self.Warning("No prior training was found under: " + pth + ". Generating...")
        self.Epoch = max([int(i) for i in f if self.IsFile(pth + "/" + i + "/TorchSave.pth")])
        self.Model.Epoch = self.Epoch
        self.Model.load 
        self._op.Epoch = self.Epoch 
        self._op.load 

    @property
    def _searchdatasplits(self):
        pth = self.OutputDirectory + "/DataSets/"
        ls = self.ls(pth)
        if len(ls) == 0: return self.Warning("No sample splitting found (k-Fold). Trainig on entire sample.") 
        if self.TrainingName + ".pkl" in ls: 
            f = UnpickleObject(pth + self.TrainingName)
            if self.kFold == None: self.kFold = iter([f[i] for i in f])
            else: self.kFold = f[self.kFold]
            return self.Success("Found the training sample: " + self.TrainingName)
        self.Warning("The given training sample name was not found, but found the following: " + "\n-> ".join(ls))
