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
        kf = []
        if not self.ContinueTraining: return True
        if len(f) == 0: return self.Warning("No prior training was found under: " + pth + ". Generating...")
        
        kfolds = list(self.kFold) 
        for epochs in f: 
            _kf = {k for k in self.ls(pth + "/" + epochs) if self.IsFile(pth + "/" + epochs + "/" + k + "/TorchSave.pth") and k in kfolds}
            if len(_kf) == 0: continue
            self.Epoch = int(epochs) if int(epochs) > self.Epoch else self.Epoch
            kf = _kf

        epochs = str(self.Epoch)
        for i in kf:
            self._kModels[i]._pth = self.OutputDirectory + "/" + self.RunName 
            self._kModels[i].Epoch = epochs + "/" + i 
            try: self._kModels[i].load
            except KeyError: return not self.Failure("Loading Model Failed. Exiting...")
            self._kOp[i]._pth = self.OutputDirectory + "/" + self.RunName 
            self._kOp[i].Epoch = epochs + "/" + i 
            try: self._kOp[i].load
            except KeyError: return not self.Failure("Loading Optimizer Failed. Exiting...")
        for i in list(self._kModels):
            if i in kf: continue
            del self._kModels[i] 
            del self._kOp[i]
            del self._DataLoader[i]
            self.Warning("Removing " + i + " from training session. Due to inconsistent epoch.")

    @property
    def _searchdatasplits(self):
        pth = self.OutputDirectory + "/DataSets/"
        ls = self.ls(pth)
        if len(ls) == 0: return self.Warning("No sample splitting found (k-Fold). Training on entire sample.") 
        if self.TrainingName + ".pkl" in ls: 
            f = UnpickleObject(pth + self.TrainingName)
            if self.kFold == None: self.kFold = f
            elif isinstance(self.kFold, int) and "k-" + str(self.kFold) in f: self.kFold = {"k-" + str(self.kFold) : f["k-" + str(self.kFold)]}
            elif "k-" + str(self.kFold) in f: self.kFold = {self.kFold : f[self.kFold]}
            else: 
                self.Warning("Given k-Fold not found. Assuming the following folds: " + "\n-> ".join([""] + list(f)))
                self.kFold = f
            return self.Success("Found the training sample: " + self.TrainingName)
        self.Warning("The given training sample name was not found, but found the following: " + "\n-> ".join(ls))

    @property
    def _showloss(self): 
        string = ["Epoch-kFold: " + self.Epoch]
        if self._op._sc is not None: string[-1] += " Current LR: {:.10f}".format(self._op._sc.get_lr()[0])
        for f in self.Model._l: 
            string.append("Feature: {}, Loss: {:.10f}, Accuracy: {:.10f}".format(f, self.Model._l[f]["loss"], self.Model._l[f]["acc"]))
        print("\n-> ".join(string)) 

