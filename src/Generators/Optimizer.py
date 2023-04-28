from AnalysisG.Notification import _Optimizer
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from AnalysisG.Model import ModelWrapper
from AnalysisG.Tools import Code 

class Optimizer(_Optimizer, Settings, SampleTracer):

    def __init__(self, inpt):
        self.Caller = "OPTIMIZER"
        Settings.__init__(self) 
        SampleTracer.__init__(self)
        _Optimizer.__init__(self)  
        if issubclass(type(inpt), SampleTracer): self += inpt
        if issubclass(type(inpt), Settings): self.ImportSettings(inpt)

    @property
    def Launch(self):
        self.DataCache = True 
        if self._NoModel: return False
        if self._NoSampleGraph: return False
        self._Code["Model"] = Code(self.Model)
        self.Model = ModelWrapper(self._Code["Model"].clone)
        
        for i in self: break
        if not self.Model.SampleCompatibility(i): return self._notcompatible
        pred, loss = self.Model(i)
        #print(pred, loss)
        


