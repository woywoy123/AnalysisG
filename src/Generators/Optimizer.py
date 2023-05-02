from AnalysisG.Model.Optimizers import OptimizerWrapper
from AnalysisG.Notification import _Optimizer
from AnalysisG.Evaluation.Epoch import Epoch
from .SampleGenerator import RandomSamplers
from AnalysisG.Tools import Code, Threading
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Model import ModelWrapper
from AnalysisG.Settings import Settings
from torch_geometric.data import Batch
from .Interfaces import _Interface
from multiprocessing import Process

class Optimizer(_Optimizer, _Interface, SampleTracer, RandomSamplers):

    def __init__(self, inpt):
        self.Caller = "OPTIMIZER"
        Settings.__init__(self) 
        SampleTracer.__init__(self)
        _Optimizer.__init__(self)  
        if issubclass(type(inpt), SampleTracer): self += inpt
        if issubclass(type(inpt), Settings): self.ImportSettings(inpt)

    @property
    def __this_epoch__(self):
        if self._len== 0: return 
        if not self.DebugMode:
            title = "(Training) " if self.Model.train else "(Validation) "
            title += "Epoch " + str(self.Epoch).split("/")[0] + "/" + str(self.Epochs) + " k-Fold: " + str(self.Epoch).split("/")[1]
            _, bar = self._MakeBar(self._len, title)          
        for smpl in self._dl: 
            self._op.zero
            self._ep.start
            pred, loss = self.Model(smpl)
            self._ep.end
            self._ep.Collect(smpl, pred, loss)
            self.Model.backward
            self._op.step
            
            if not self.DebugMode: bar.update(1)
            if self.DebugMode: self._showloss
            if not self.EnableReconstruction: continue
            self.Model.TruthMode = True 
            truth = self.Model.mass
            self.Model.TruthMode = False 
            pred = self.Model.mass
            eff = self.Model.ParticleEfficiency 
            dc = self._ep._train if self._ep.train else self._ep._val
            for b in range(len(truth)):
                for f in truth[b]: dc[f]["mass_t"] += truth[b][f]
                for f in pred[b]: dc[f]["mass"] += pred[b][f]
                for f in eff[b]: dc[f]["nrec"].append(eff[b][f]["nrec"])       
                for f in eff[b]: dc[f]["ntru"].append(eff[b][f]["ntru"])

    def __dump_plots__(self, dumper, end = False):
        if len(dumper) == self.Threads and not end: return dumper
        def func(inpt): 
            for i in inpt: i.dump 

        th = Process(target = func, args = (dumper, ))
        th.start()
        return [] 

    @property 
    def __train__(self):
        dumper = []
        for i in range(self.Epoch, self.Epochs):
            for k in self._kModels:
                self.Epoch = str(i+1) + "/" + str(k)
                self.mkdir(self.OutputDirectory + "/" + self.RunName + "/" + str(self.Epoch))
                
                self.Model = self._kModels[k] 
                self.Model.Epoch = str(i+1) + "/" + str(k)
                self._op = self._kOp[k]
                self._DL = iter(self._DataLoader[k])

                self._ep = Epoch()
                self._ep.o_model = self.Model.o_mapping 
                self._ep.i_model = self.Model.i_mapping 
                self._ep.RunName = self.RunName 
                self._ep.OutputDirectory = self.OutputDirectory
                self._ep.init
                self._op.Epoch = str(i+1) + "/" + str(k)
                self._ep.Epoch = str(i+1) + "/" + str(k)
               
                mode = next(self._DL) 
                self._dl = self._DataLoader[k][mode]
                self._len = self._nsamples[k][mode]
                self.Model.train = True
                self._ep.train = True
                self._op.train = True
                self.__this_epoch__
 
                try: 
                    mode = next(self._DL)
                    self._dl = self._dl = self._DataLoader[k][mode]
                    self._len = self._nsamples[k][mode]
                except StopIteration: self._len = 0 
                
                self.Model.train = False
                self._ep.train = False
                self._op.train = False
                self.__this_epoch__
                
                self._op.stepsc
                self._op.dump
                self.Model.dump
                dumper.append(self._ep)
                dumper = self.__dump_plots__(dumper)
        dumper = self.__dump_plots__(dumper, True)         
    @property 
    def GetCode(self):
        if "Model" in self._Code: return self._Code["Model"]
        self._Code["Model"] = Code(self.Model)
        return self.GetCode
 
    @property
    def Launch(self):

        self.DataCache = True 
        if self._NoModel: return False
        if self._NoSampleGraph: return False
        self.OutputDirectory += "/Training"
        self.Model = ModelWrapper(self.GetCode.clone)
        self.Model.OutputDirectory = self.OutputDirectory
        self.Model.RunName = self.RunName
        
        for smpl in self: break
        if not self.Model.SampleCompatibility(smpl): return self._notcompatible
        
        self._op = OptimizerWrapper()
        self._op.ImportSettings(self)
        self._op._mod = self.Model._Model
        if self._setoptimizer: return 
        self._setscheduler
        self._searchdatasplits
        self._op = None
        self.Model = None

        self._kModels = {}
        self._kOp = {}
        self._DataLoader = {}
        self._nsamples = {}
        if self.kFold is None: self.kFold = {"train" : {"all" : [list(self.todict)]}}
        for k in self.kFold:
            self._kModels[k] = ModelWrapper(self._Code["Model"].clone)
            self._kModels[k].OutputDirectory = self.OutputDirectory
            self._kModels[k].RunName = self.RunName
            self._kModels[k].SampleCompatibility(smpl)
            self._kModels[k].device = self.Device
 
            self._kOp[k] = OptimizerWrapper()
            self._kOp[k].ImportSettings(self)
            self._kOp[k]._mod = self._kModels[k]._Model
            self._kOp[k].SetOptimizer
            self._kOp[k].SetScheduler
            self._nsamples[k] = {}
            self._DataLoader[k] = {}
            for s in self.kFold[k]: 
                self.MarkTheseHashes(self.kFold[k][s], s)
                self.ForceTheseHashes(self.kFold[k][s])
                self._DataLoader[k][s] = self.MakeDataLoader([i.clone().to(self.Device) for i in self], self.SortByNodes, self.BatchSize)
                self._DataLoader[k][s] = [Batch().from_data_list(t) for i in self._DataLoader[k][s].values() for t in i]
                self._nsamples[k][s] = len(self._DataLoader[k][s])
        if not self._searchtraining: return
        self.__train__
