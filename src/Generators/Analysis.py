from AnalysisG.Generators.SampleGenerator import RandomSamplers
from AnalysisG.Generators.EventGenerator import EventGenerator
from AnalysisG.Generators.GraphGenerator import GraphGenerator
from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Templates import FeatureAnalysis
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Notification import _Analysis
from typing import Union

#from AnalysisG.IO import PickleObject, UnpickleObject
#from .SelectionGenerator import SelectionGenerator
#from .GraphGenerator import GraphGenerator
#from AnalysisG.Settings import Settings
#from .Optimizer import Optimizer


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
        self.TestFeatures = False
        self.triggered = False
        self.nEvents = 10
        self.TrainingSize = False
        self.kFolds = False
        self.TrainingName = "untitled"

        if Name is None and SampleDirectory is None: return
        self.InputSample(Name, SampleDirectory)





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
        pth = self.WorkingDirectory + "Training/DataSets/"
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






    def __build__(self):
        self.StartingAnalysis()
        self._BuildingCache()
        if self.PurgeCache: self._WarningPurge()
        self.triggered = True

    def __scan__(self):
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
        return f

    def __FeatureAnalysis__(self):
        if self.Graph is None: return True
        if not self.TestFeatures: return

        self.GetAll = True
        tests = self.makelist()
        self.GetAll = False
        if len(tests) < self.nEvents: tests
        else: tests = tests[:self.nEvents]

        code_ = self.Graph.code
        code  = code_["__state__"]

        f = FeatureAnalysis()
        f.Verbose = self.Verbose
        for gr, i in code["graph_feature"].items():
            gr, i = gr.decode("UTF-8")[4:], i.decode("UTF-8")
            f.GraphAttribute[gr] = code_[i]

        for no, i in code["node_feature"].items():
            no, i = no.decode("UTF-8")[4:], i.decode("UTF-8")
            f.NodeAttribute[no] = code_[i]

        for ed, i in code["edge_feature"].items():
            ed, i = ed.decode("UTF-8")[4:], i.decode("UTF-8")
            f.EdgeAttribute[ed] = code_[i]

        return f.TestEvent(tests, self.Graph)

    def __Graph__(self):
        if self.Graph is None: return True
        if not len(self.ShowLength): self.RestoreEvents()
        if not len(self.ShowLength): return True

        gr = GraphGenerator(self)
        gr.ImportSettings(self.ExportSettings())
        gr.Caller = "ANALYSIS::GRAPH"
        if not gr.MakeGraphs(): return False
        self += gr

        if not self.DataCache: return True
        gr.DumpGraphs()
        gr.DumpTracer(self.SampleName)
        return True

    def __Event__(self):
        if self.Event is None: return True

        self.GetEvent = True
        self.GetGraphs = False
        f = self.__scan__()
        if not len(f): return True
        self.Files = None
        self.Files = f

        if self.EventStop is None: pass
        elif len(self) >= self.EventStop: return True
        ev = EventGenerator()
        ev.Caller = "ANALYSIS::EVENT"
        ev.ImportSettings(self.ExportSettings())
        if self.EventStop is not None: ev.EventStop -= len(self)
        if not ev.MakeEvents(self.SampleName): return False
        self += ev

        if not self.EventCache: return True
        ev.DumpEvents()
        ev.DumpTracer(self.SampleName)
        return True

    def __LoadSample__(self):
        tracer = self._CheckForTracer()
        for name in self.SampleMap:
            if not len(name): name = None
            if not tracer: break
            self.RestoreTracer(tracer, name)

        l = len(self.SampleMap)
        if not l: self.RestoreTracer()
        if self.EventCache: self.RestoreEvents()
        if self.DataCache: self.RestoreGraphs()
        for name in self.SampleMap:
            self.SampleName = name
            self.__Event__()
            self.__FeatureAnalysis__()
            self.__Graph__()
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
        content = self.ShowLength
        if not len(content): return self.EmptySampleList()
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: pass

        if not len(self.EventName):
            try:
                self.EventName = self.ShowEvents[0]
                content[self.Tree + "/" + self.EventName]
                self.GetEvent = True
            except IndexError: self.GetEvent = False
            except KeyError: self.EventName = ""

        if not len(self.GraphName):
            try:
                self.GraphName = self.ShowGraphs[0]
                content[self.Tree + "/" + self.GraphName]
                self.GetGraph = True
            except IndexError: self.GetGraph = False
            except KeyError: self.GraphName = ""

        if not len(self.SelectionName):
            try:
                self.SelectionName = self.ShowSelections[0]
                self.GetSelection = True
            except IndexError: self.GetSelection = False
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: pass

        return False



