from AnalysisG.IO import PickleObject, UnpickleObject
from .SelectionGenerator import SelectionGenerator 
from AnalysisG.Templates import FeatureAnalysis
from AnalysisG.Notification import _Analysis
from .SampleGenerator import RandomSamplers
from .EventGenerator import EventGenerator 
from .GraphGenerator import GraphGenerator 
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from .Interfaces import _Interface
from .Optimizer import Optimizer

class Analysis(_Analysis, Settings, SampleTracer, _Interface):
    
    def __init__(self):
        self.Caller = "ANALYSIS"
        _Analysis.__init__(self)
        _Interface.__init__(self)
        Settings.__init__(self)
        SampleTracer.__init__(self)
  
    @property
    def __build__(self):
        
        if self._cPWD is not None: return 
        if not self._condor: self.StartingAnalysis
        self._cPWD = self.pwd
        if self.OutputDirectory is None: self.OutputDirectory = self.pwd
        else: self.OutputDirectory = self.abs(self.OutputDirectory)
        self.OutputDirectory = self.AddTrailing(self.OutputDirectory, "/") + self.ProjectName
        if self.PurgeCache: self._WarningPurge
        if not self._condor: self._BuildingCache 

    @property
    def __Selection__(self):
        if len(self.Selections) == 0 and len(self.Merge) == 0: return 
        tmp = self.DataCache
        self.DataCache = False
        self.EventCache = True
        if not self.__LoadSample__: return False

        pth = self.OutputDirectory + "/Selections/"
        sel = SelectionGenerator(self)
        sel.ImportSettings(self)
        sel.Caller = "ANALYSIS::SELECTIONS"
        sel.MakeSelection
        del sel
        self.DataCache = tmp
 
    @property 
    def __Event__(self):
        process = {}

        for i in list(self.Files): 
            f = [j for j in self.Files[i] if i + "/" + j not in self]
            if len(f) != 0: process[i] = f
        if len(process) == 0: return True 
        if self.Event == None: return False
        if self.EventStop != None and len(self) >= self.EventStop: return True 
        self.Files = process 
        ev = EventGenerator()
        ev.ImportSettings(self)
        ev.Caller = "ANALYSIS::EVENT"
        if not ev.MakeEvents: return False
        if self.EventCache: ev.DumpEvents
        self += ev
        return True 

    @property
    def __Graph__(self):
        if self.EventGraph == None: return True
        process = {}
        for i in list(self.Files): 
            f = {j : len([l for l in self[i + "/" + j] if l.Event]) for j in self.Files[i]}
            f = [j for j in f if j != 0] 
            if len(f) != 0: process[i] = f
        if len(process) == 0 and len(self.Files) != 0: return True 
        failed = False
        if self.TestFeatures: failed = self.__FeatureAnalysis__
        if failed: return False 

        self.Files = process 
        gr = GraphGenerator(self)
        gr.ImportSettings(self)
        gr.Caller = "ANALYSIS::GRAPH"
        if not gr.MakeGraphs: return False
        self += gr
        if self.DataCache: self.DumpEvents
        return True 
   
    @property
    def __FeatureAnalysis__(self):
        if self.EventGraph is None: return True
        if not self.TestFeatures: return 

        f = FeatureAnalysis()
        f.ImportSettings(self)
        return f.TestEvent([i for i, _ in zip(self, range(self.nEvents))], self.EventGraph) 
 
    @property 
    def __RandomSampler__(self):
        pth = self.OutputDirectory + "/Training/DataSets/" 
        if not self.TrainingSize and not self.kFolds: return 
        if self.TrainingName + ".pkl" in self.ls(pth): return 

        self.EventCache = False
        self.DataCache = True
        if not self.__LoadSample__: return False
 
        r = RandomSamplers()
        r.Caller = self.Caller
        output = {}
        if self.TrainingSize: output = r.MakeTrainingSample(self.todict, self.TrainingSize) 
        if self.kFolds: output |= r.MakekFolds(self.todict, self.kFolds, self.BatchSize, self.Shuffle, True)
        if len(output) == 0: return  
        self.mkdir(pth)
        PickleObject(output, pth + self.TrainingName)

    @property 
    def __Optimizer__(self):
        if self.Model == None and self.Optimizer == None: return
        
        self.EventCache = False
        self.DataCache = True
        if not self.__LoadSample__: return False
       
        op = Optimizer(self)
        op.Launch

    @property
    def __LoadSample__(self):
        self.__build__
        tracer = self._CheckForTracer
        for i in self.SampleMap:
            self.Files = self.SampleMap[i]
            self.SampleName = i
            if tracer: self.RestoreEvents
            if not self.__Event__: return False
            if not self.__Graph__: return False
        if len(self.SampleMap) == 0: self.RestoreEvents
        if self.EventCache: self.RestoreEvents
        if self.DataCache: self.RestoreEvents
        if len(self) == 0: return False
        else: return True 
 
    @property
    def __CollectCode__(self):
        code = {}
        if self.Event is not None:
            ev = EventGenerator()
            ev.ImportSettings(self)
            code |= ev.MakeEvents
        
        if self.EventGraph is not None:
            gr = GraphGenerator()
            gr.ImportSettings(self)
            code |= gr.MakeGraphs
        
        if len(self.Selections) != 0:
            sel = SelectionGenerator(self)
            sel.ImportSettings(self)
            for name in self.Selections: sel.AddSelection(name, self.Selections[name])
            code |= sel.MakeSelection
        if self.Model is not None: code["Model"] = Optimizer(self).GetCode
        return code
    
    @property
    def Launch(self):   
        if self._condor: return self.__CollectCode__
        self.__build__
        self.__Selection__
        self.__RandomSampler__
        self.__Optimizer__
        self.WhiteSpace()
        if self.EmptySampleList: return self.__LoadSample__ 
        return True
    
    def __preiteration__(self):
        if len(self) == 0: return not self.__LoadSample__
        return self.EmptySampleList
 
