from AnalysisG.Generators.EventGenerator import EventGenerator
from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Notification import _Analysis
#from AnalysisG.IO import PickleObject, UnpickleObject
#from .SelectionGenerator import SelectionGenerator
#from AnalysisG.Templates import FeatureAnalysis
#from .SampleGenerator import RandomSamplers
#from .GraphGenerator import GraphGenerator
#from AnalysisG.Settings import Settings
#from .Optimizer import Optimizer
from typing import Union


class Analysis(_Analysis, SampleTracer, _Interface):
    def __init__(
        self,
        SampleDirectory: Union[str, dict, list, None] = None,
        Name: Union[str] = None,
    ):
        SampleTracer.__init__(self)
        self.Caller = "ANALYSIS"
        _Analysis.__init__(self)
        _Interface.__init__(self)
        self.PurgeCache = False
        self.triggered = False
        if Name is None and SampleDirectory is None: return
        self.InputSample(Name, SampleDirectory)

    def __build__(self):
        self.StartingAnalysis()
        self._BuildingCache()
        if self.PurgeCache: self._WarningPurge()
        self.triggered = True

    def __Graph__(self):
        if self.EventGraph == None: return True

        process = {}
        if self.EventCacheLen != self.DataCacheLen:
            process.update(self.Files)
        if len(process) == 0 and len(self.Files) != 0:
            return True
        self.RestoreEvents()
        failed = False
        if self.TestFeatures: failed = self.__FeatureAnalysis__()
        if failed: return False

        gr = GraphGenerator(self)
        gr.ImportSettings(self)
        gr.Caller = "ANALYSIS::GRAPH"
        if not gr.MakeGraphs(): return False

        gr.DataCache = self.DataCache
        if self.DataCache: gr.DumpEvents()
        self += gr
        return True

    def __FeatureAnalysis__(self):
        if self.EventGraph is None: return True
        if not self.TestFeatures: return
        tests = [i for i, _ in zip(self.GetEventCacheHashes(), range(self.nEvents))]
        self.ForceTheseHashes(tests)
        f = FeatureAnalysis()
        f.ImportSettings(self)
        return f.TestEvent([i for i in self], self.EventGraph)

    def __Selection__(self):
        if len(self.Selections) == 0 and len(self.Merge) == 0:
            return
        if self.EventCacheLen == 0: return
        self.EventCache = True
        self.RestoreEvents()

        pth = self.OutputDirectory + "/Selections/"
        sel = SelectionGenerator(self)
        sel.Threads = self.Threads # Fix after merge
        sel.ImportSettings(self)
        sel.Caller = "ANALYSIS::SELECTIONS"
        sel.MakeSelection()
        del sel

    def __RandomSampler__(self):
        pth = self.OutputDirectory + "/Training/DataSets/"
        if not self.TrainingSize: return
        if self.TrainingName + ".pkl" in self.ls(pth): return
        if not self.kFolds: self.kFolds = 1

        output = {}
        r = RandomSamplers()
        r.Caller = self.Caller
        if self.TrainingSize:
            output = r.MakeTrainingSample(self.todict(), self.TrainingSize)
        if self.kFolds:
            output.update(
                r.MakekFolds(
                    self.todict(), self.kFolds, self.BatchSize, self.Shuffle, True
                )
            )
        if len(output) == 0: return
        self.mkdir(pth)
        PickleObject(output, pth + self.TrainingName)

    def __Optimizer__(self):
        if self.Model == None and self.Optimizer == None:
            return
        op = Optimizer(self)
        op.Launch()



    def __Event__(self):
        if self.Event is None: return True
        f = {}
        self.GetAll = True
        for i in self.SampleMap[self.SampleName]:
            i = self.abs(i)
            if self.SampleName is None and i in self:
                continue
            if self.SampleName not in self: pass
            elif i in self: continue
            i = i.split("/")
            file, path = i[-1], "/".join(i[:-1])
            if path not in f: f[path] = []
            f[path].append(file)
        self.GetAll = False
        if not len(f): return True
        self.Files = None
        self.Files = f

        if self.EventStop is None: pass
        elif len(self) >= self.EventStop: return True
        ev = EventGenerator()
        ev.Caller = "ANALYSIS::EVENT"
        ev.ImportSettings(self.ExportSettings())
        if self.EventStop is not None: ev.EventStop -= len(self)
        if not ev.MakeEvents(): return False
        ev.EventStop = None
        self += ev
        if not self.EventCache: return True
        self.DumpEvents()
        self.DumpTracer(self.SampleName)
        return True

    def __LoadSample__(self):
        tracer = self._CheckForTracer()
        for name in self.SampleMap:
            if not len(name): name = None
            if not tracer: break
            self.RestoreTracer(tracer, name)
        if self.EventCache: self.RestoreEvents()
        for name in self.SampleMap:
            self.SampleName = name
            self.__Event__()
            #if not self.__Graph__(): return False
        #if len(self) == 0: return False
        return True

    def Launch(self):
        self.__build__()
        self.__LoadSample__()
        #self.__Selection__()
        #self.__RandomSampler__()
        #self.__Optimizer__()
        #self.WhiteSpace()
        return True

    def preiteration(self) -> bool:
        if not self.triggered: self.Launch()
        if not len(self.EventName):
            try:
                self.EventName = self.ShowEvents[0]
                self.GetEvent = True
            except IndexError: self.GetEvent = False

        if not len(self.GraphName):
            try:
                self.GraphName = self.ShowGraphs[0]
                self.GetGraph = True
            except IndexError: self.GetGraph = False

        if not len(self.SelectionName):
            try:
                self.SelectionName = self.ShowSelections[0]
                self.GetSelection = True
            except IndexError: self.GetSelection = False
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: return True

        return False



