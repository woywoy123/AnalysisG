from AnalysisG.Generators.SampleGenerator import RandomSamplers
from AnalysisG._cmodules.cWrapping import OptimizerWrapper
from AnalysisG._cmodules.cOptimizer import cOptimizer
from AnalysisG.Notification import _Optimizer
from .Interfaces import _Interface
from AnalysisG.Model import Model
import h5py

class Settings:

    def __init__(self):
        self.TrainingName = "untitled"

        self.Model = None
        self.ModelParams = None
        self.RunName = "run_name"

        self.kFolds = 10
        self.kFold = None
        self.Device = "cpu"

        self.Epochs = 20
        self.Epoch = None

        self.Optimizer = None
        self.OptimizerParams = {}

        self.Scheduler = None
        self.SchedulerParams = {}

        self.BatchSize = 1
        self.DebugMode = None
        self.ContinueTraining = False
        self.SortByNodes = True
        self.EnableReconstruction = True


class Optimizer(_Optimizer, _Interface, RandomSamplers):
    def __init__(self, inpt = None):
        RandomSamplers.__init__(self)
        self.Caller = "OPTIMIZER"
        Settings.__init__(self)
        _Optimizer.__init__(self)
        self._kOps = {}
        self._kModels = {}
        self._cmod = cOptimizer()
        self.triggered = False
        if inpt is None: return
        if not self.is_self(inpt): return
        else: self += inpt

    def __searchsplits__(self):
        sets = self.WorkingPath + "Training/DataSets/"
        sets += self.TrainingName
        if not self._cmod.GetHDF5Hashes(sets):
            self.GetAll = True
            self._cmod.UseAllHashes(self.makehashes())
            self.GetAll = False
        self._searchdatasplits(sets)

    def __initialize__(self):
        for k in self._cmod.kFolds:
            batches = self._cmod.FetchTraining(k, self.BatchSize)
            graphs = self._cmod.MakeBatch(self, next(iter(batches)), k, 0)

            self._kModels[k] = Model(self.Model)
            self._kModels[k].Path = self.WorkingPath + "machine-learning/"
            self._kModels[k].RunName = self.RunName
            self._kModels[k].KFold = k

            self._kModels[k].__params__ = self.ModelParams
            self._kModels[k].device = self.Device
            if not self._kModels[k].SampleCompatibility(graphs): return

            self._kOps[k] = OptimizerWrapper()
            self._kOps[k].Path = self.WorkingPath + "machine-learning/"
            self._kOps[k].RunName = self.RunName
            self._kOps[k].KFold = k

            self._kOps[k].Optimizer = self.Optimizer
            self._kOps[k].OptimizerParams = self.OptimizerParams

            self._kOps[k].Scheduler = self.Scheduler
            self._kOps[k].SchedulerParams = self.SchedulerParams
            self._kOps[k].model = self._kModels[k].model

            if not self._kOps[k].setoptimizer(): return
            self._kOps[k].setscheduler()

    def Start(self, sample = None):
        if sample is None: pass
        else: self.ImportSettings(sample.ExportSettings())
        self.RestoreTracer()
        self.__searchsplits__()
        self._findpriortraining()
        self.__initialize__()
        kfolds = self._cmod.kFolds
        kfolds.sort()

        path = self.WorkingPath + "machine-learning/" + self.RunName
        for ep in range(self.Epoch, self.Epochs):
            ep += 1
            for k in kfolds:
                self.mkdir(path + "/Epoch-" + str(ep) + "/kFold-" + str(k))

                self._kOps[k].Epoch = ep
                self._kModels[k].Epoch = ep

                self._kOps[k].Train = True
                self._kModels[k].Train = True
                self.__train__(k, ep)

                self._kOps[k].Train = False
                self._kModels[k].Train = False
                self.__validation__(k, ep)

                self._kOps[k].stepsc()

    def __train__(self, kfold, epoch):
        batches = self._cmod.FetchTraining(kfold, self.BatchSize)
        index_ = len(batches)
        _, bar = self._MakeBar(index_, "Epoch (" + str(epoch) + ") Running k-Fold (training) -> " + str(kfold))
        for i, batch in zip(range(index_), batches):
            graph = self._cmod.MakeBatch(self, batch, kfold, i)
            self._kOps[kfold].zero()

            res = self._kModels[kfold](graph)
            self._kModels[kfold].backward()

            self._kOps[kfold].step()
            bar.update(1)

    def __validation__(self, kfold, epoch):
        batches = self._cmod.FetchValidation(kfold, self.BatchSize)
        index_ = len(batches)
        _, bar = self._MakeBar(index_, "Epoch (" + str(epoch) + ") Running k-Fold (validation) -> " + str(kfold))
        for i, batch in zip(range(index_), batches):
            graph = self._cmod.MakeBatch(self, batch, kfold, i)
            bar.update(1)



    def preiteration(self, inpt = None):
        if self.triggered: return False

        if inpt is not None: pass
        else: inpt = self
        if len(self.GraphName): pass
        elif not len(self.GraphName):
            try: self.GraphName = inpt.ShowGraphs[0]
            except IndexError: return self._nographs()

        if len(self.Tree): return False
        try: self.Tree = inpt.ShowTrees[0]
        except IndexError: return self._nographs()

        self.Success("Using GraphName: " + self.GraphName + " with Tree: " + self.Tree)
        self.triggered = True
        inpt.Tree = self.Tree
        inpt.GraphName = self.GraphName
        inpt.GetGraph = True
        return False

