from AnalysisG.Model.Optimizers import OptimizerWrapper
from AnalysisG.Notification import _Optimizer
from AnalysisG.Evaluation.Epoch import Epoch
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from AnalysisG.Model import ModelWrapper
from AnalysisG.Tools import Code
from .Interfaces import _Interface

from time import sleep

class Optimizer(_Optimizer, Settings, _Interface, SampleTracer):

    def __init__(self, inpt):
        self.Caller = "OPTIMIZER"
        Settings.__init__(self) 
        SampleTracer.__init__(self)
        _Optimizer.__init__(self)  
        if issubclass(type(inpt), SampleTracer): self += inpt
        if issubclass(type(inpt), Settings): self.ImportSettings(inpt)

    @property
    def __this_epoch__(self):
        bar = None
        for smpl in self: 
            if not self.DebugMode and bar is None:
                title = "(Training) " if self.Model.train else "(Validation) "
                title += "Epoch " + str(self.Epoch) + "/" + str(self.Epochs)
                _, bar = self._MakeBar(next(self._nsamples), title)
            
            print(smpl.TrainMode)
            self._op.zero
            self._ep.start
            pred, loss = self.Model(smpl)
            self._ep.end
            self._ep.Collect(smpl, pred, loss)
            self.Model.backward
            self._op.step
            if not self.DebugMode: bar.update(1)
            sleep(0.1)
           
    @property 
    def __train__(self):
        for i in range(self.Epoch, self.Epochs):
            self._ep = Epoch()
            self._ep.o_model = self.Model.o_mapping 
            self._ep.i_model = self.Model.i_mapping 
            self._ep.RunName = self.RunName 
            self._ep.OutputDirectory = self.OutputDirectory
            self._ep.init
            self.Epoch = i+1
            self.Model.Epoch = i+1
            self._op.Epoch = i+1
            self._ep.Epoch = i+1
 
            pth = self.OutputDirectory + "/" + self.RunName + "/" + str(self.Epoch) + "/"
            self.mkdir(pth)

            self.Model.train = True
            self._ep.train = True
            self._op.train = True
            self.__this_epoch__

            self.Model.train = False
            self._ep.train = False
            self._op.train = False
            self.__this_epoch__
            
            self._op.stepsc
            self._op.dump
            self._ep.dump
            self.Model.dump

    @property
    def Launch(self):
        self.DataCache = True 
        if self._NoModel: return False
        if self._NoSampleGraph: return False
        self.OutputDirectory += "/Training"
        self._Code["Model"] = Code(self.Model)
        self.Model = ModelWrapper(self._Code["Model"].clone)
        self.Model.OutputDirectory = self.OutputDirectory
        self.Model.RunName = self.RunName
        
        for i in self: break
        if not self.Model.SampleCompatibility(i): return self._notcompatible
        
        self._op = OptimizerWrapper()
        self._op.ImportSettings(self)
        if self._setoptimizer: return 
        self._setscheduler
        self._searchtraining
        self._searchdatasplits
        self.__train__

    def __preiteration__(self):
        if self.kFold is None: return 
        it = next(self.kFold)
        hashes = {i : [t for j in it[i] for t in j] for i in it}
        hashes_l = []
        for i in hashes: 
            self.MarkTheseHashes(hashes[i], i)
            hashes_l += hashes[i]
        self.ForceTheseHashes(hashes_l)
        self._nsamples = iter([len(hashes[i]) for i in hashes])
