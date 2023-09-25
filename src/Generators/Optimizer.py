from AnalysisG.Generators.SampleGenerator import RandomSamplers
from AnalysisG._cmodules.cWrapping import OptimizerWrapper
from AnalysisG._cmodules.cOptimizer import cOptimizer
from AnalysisG.Notification import _Optimizer
from .Interfaces import _Interface
from AnalysisG.Model import Model
from torchmetrics import ROC, AUROC
import torch
import h5py

class Optimizer(_Optimizer, _Interface, RandomSamplers):
    def __init__(self, inpt = None):
        RandomSamplers.__init__(self)
        self.Caller = "OPTIMIZER"
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
        if self._nomodel(): return False
        kfolds = self._cmod.kFolds
        if len(kfolds): pass
        else: self._cmod.UseTheseFolds([1])
        for k in self._cmod.kFolds:
            batches = self._cmod.FetchTraining(k, self.BatchSize)
            graphs = self._cmod.MakeBatch(self, next(iter(batches)), k, 0)

            self._kModels[k] = Model(self.Model)
            self._kModels[k].Path = self.WorkingPath + "machine-learning/"
            self._kModels[k].RunName = self.RunName
            self._kModels[k].KFold = k
            self._kModels[k].__params__ = self.ModelParams
            self._kModels[k].device = self.Device
            if self._kModels[k].SampleCompatibility(graphs): pass
            else: return self._notcompatible()

            self._kOps[k] = OptimizerWrapper()
            self._kOps[k].Path = self.WorkingPath + "machine-learning/"
            self._kOps[k].RunName = self.RunName
            self._kOps[k].KFold = k

            self._kOps[k].Optimizer = self.Optimizer
            self._kOps[k].OptimizerParams = self.OptimizerParams

            self._kOps[k].Scheduler = self.Scheduler
            self._kOps[k].SchedulerParams = self.SchedulerParams
            self._kOps[k].model = self._kModels[k].model

            if self._kOps[k].setoptimizer(): pass
            else: return self._invalidoptimizer()

            if self._kOps[k].setscheduler(): pass
            else: return self._invalidscheduler()
        return True

    def Start(self, sample = None):
        if sample is None: pass
        else: self.ImportSettings(sample.ExportSettings())

        self.RestoreTracer()
        self.__searchsplits__()
        if self.__initialize__(): pass
        else: return
        self._findpriortraining()
        kfolds = self._cmod.kFolds
        kfolds.sort()
        self._cmod.metric_plot.plot = self.PlotLearningMetrics
        path = self.WorkingPath + "machine-learning/" + self.RunName
        for ep in range(self.Epochs):
            ep += 1
            kfold_map = {}
            for k in kfolds:
                base = path + "/Epoch-" + str(ep) + "/kFold-"
                if self.Epoch is None: pass
                elif k not in self.Epoch: pass
                elif self.Epoch[k] < ep: kfold_map[k] = False
                else:
                    kfold_map[k] = True
                    self._cmod.RebuildEpochHDF5(ep, base, k)
                    continue
                self.mkdir(base + str(k))
                self._kOps[k].Epoch = ep
                self._kModels[k].Epoch = ep

                self._kOps[k].Train = True
                self._kModels[k].Train = True
                self.__train__(k, ep)
                self._showloss(ep, k)

                self._kOps[k].Train = False
                self._kModels[k].Train = False
                self.__validation__(k, ep)
                self._kOps[k].stepsc()

                self.__evaluation__(k, ep)

                self._kOps[k].save()
                self._kModels[k].save()
            if sum(kfold_map.values()) == len(kfold_map): pass
            else: self._cmod.DumpEpochHDF5(ep, base, kfolds)
            x = self._cmod.BuildPlots(ep, path)

    def __train__(self, kfold, epoch):
        batches = self._cmod.FetchTraining(kfold, self.BatchSize)
        index_ = len(batches)
        msg = "Epoch (" + str(epoch) + ") Running k-Fold (training) -> " + str(kfold)
        _, bar = self._MakeBar(index_, msg)
        for i, batch in zip(range(index_), batches):
            graph = self._cmod.MakeBatch(self, batch, kfold, i)
            self._kOps[kfold].zero()
            res = self._kModels[kfold](graph)
            self._kModels[kfold].backward()
            self._cmod.AddkFold(epoch, kfold, res, self._kModels[kfold].out_map)

            self._kOps[kfold].step()
            bar.update(1)

    def __validation__(self, kfold, epoch):
        batches = self._cmod.FetchValidation(kfold, self.BatchSize)
        index_ = len(batches)
        msg = "Epoch (" + str(epoch) + ") Running k-Fold (validation) -> " + str(kfold)
        _, bar = self._MakeBar(index_, msg)
        for i, batch in zip(range(index_), batches):
            graph = self._cmod.MakeBatch(self, batch, kfold, i)
            res = self._kModels[kfold](graph)
            self._cmod.AddkFold(epoch, kfold, res, self._kModels[kfold].out_map)
            bar.update(1)

    def __evaluation__(self, kfold, epoch):
        batches = self._cmod.FetchEvaluation(self.BatchSize)
        index_ = len(batches)
        _, bar = self._MakeBar(index_, "Epoch (" + str(epoch) + ") Running leave-out")
        for i, batch in zip(range(index_), batches):
            graph = self._cmod.MakeBatch(self, batch, -1, i)
            res = self._kModels[kfold](graph)
            self._cmod.AddkFold(epoch, kfold, res, self._kModels[kfold].out_map)
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

        msg = "Using GraphName: " + self.GraphName
        msg += " with Tree: " + self.Tree
        self.Success(msg)
        self.triggered = True
        inpt.Tree = self.Tree
        inpt.GraphName = self.GraphName
        inpt.GetGraph = True
        return False

