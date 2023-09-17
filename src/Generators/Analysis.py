from AnalysisG.Generators.SelectionGenerator import SelectionGenerator
from AnalysisG.Generators.SampleGenerator import RandomSamplers
from AnalysisG.Generators.EventGenerator import EventGenerator
from AnalysisG.Generators.GraphGenerator import GraphGenerator
from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Templates import FeatureAnalysis
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Notification import _Analysis
from typing import Union

#from AnalysisG.IO import PickleObject, UnpickleObject
#from .GraphGenerator import GraphGenerator
#from AnalysisG.Settings import Settings
#from .Optimizer import Optimizer

def check_tree(x):
    if len(x.Tree): return True
    elif not len(x.ShowTrees): return True
    else: x.Tree = x.ShowTrees[0]
    return True

def check_event(x):
    if len(x.EventName): return True
    elif not len(x.ShowEvents): return False
    else: x.EventName = x.ShowEvents[0]
    return True

def check_graph(x):
    if len(x.GraphName): return True
    elif not len(x.ShowGraphs): return False
    else: x.GraphName = x.ShowGraphs[0]
    if x.Graph is None: return False
    return True

def check_selection(x):
    if len(x.SelectionName): return True
    elif not len(x.Selections): return False
    elif not len(x.ShowSelections): return False
    else: x.SelectionName = x.ShowSelections[0]
    return True

def check_number(x, path):
    try: l = x.ShowLength[path]
    except KeyError: l = 0

    if x.EventStop is None: return True
    elif l == 0: return True
    elif l >= x.EventStop: return False
    return True

def search(x):
    f = {}
    f["SampleName"] = False
    f["Files"] = []
    f["get"] = {}

    f["SampleName"] = x.SampleName in x
    try: f["Files"] = x.SampleMap[x.SampleName]
    except KeyError: pass

    for i in f["Files"]:
        x.GetAll = True
        i = x.abs(i)
        if i in x: continue
        i = i.split("/")
        file, path = i[-1], "/".join(i[:-1])
        if path not in f["get"]: f["get"][path] = []
        f["get"][path].append(file)
    x.GetAll = True
    f.update(x.makehashes())
    x.GetAll = False
    x._get = f

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
        self.SampleName = ""
        self._get = {}

        if Name is None and SampleDirectory is None: return
        self.InputSample(Name, SampleDirectory)

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


    def __FeatureAnalysis__(self):
        if self.Graph is None: return True
        if not self.TestFeatures: return

        tests = [i for i in self]
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


    def __RandomSampler__(self):
        if not self.TrainingSize: return
        pth = self.WorkingPath + "Training/DataSets/"
        pth += self.TrainingName
        dic = {i.hash : i for i in self}

        r = RandomSamplers()
        r.ImportSettings(self.ExportSettings())
        if not self.TrainingSize: out = {}
        else: out = r.MakeTrainingSample(dic, self.TrainingSize)
        r.SaveSets(out, pth)

        if not self.kFolds: out = {}
        else: out = r.MakekFolds(dic, self.kFolds, True, True)
        if out == False: return
        r.SaveSets(out, pth)

    def __Selection__(self):
        if not check_selection(self): return
        if check_event(self): pass
        else: self.__Event__()

        sel = SelectionGenerator()
        sel.ImportSettings(self.ExportSettings())
        sel.Caller = "ANALYSIS::SELECTIONS"
        if sel.MakeSelections(self): pass
        else: return
        self.DumpSelections()
        self.DumpTracer(self.SampleName)

    def __Graph__(self):
        if not check_graph(self): return

        get = sum(self._get["graph"].values(), [])
        if len(get) and self.DataCache: self.RestoreGraph(get)

        check_tree(self)
        path = self.Tree + "/" + self.GraphName
        if check_number(self, path): pass
        else: self.RestoreEvents()

        if len(self.Tree): pass
        else: self.RestoreEvents()

        gr = GraphGenerator()
        gr.ImportSettings(self.ExportSettings())
        gr.Caller = "ANALYSIS::GRAPH"
        gr.GetEvent = "Event"
        if gr.MakeGraphs(self): pass
        else: return

        if self.DataCache: pass
        else: return

        self.DumpGraphs()
        self.DumpTracer(self.SampleName)
        return True


    def __Event__(self):
        get = sum(self._get["event"].values(), [])
        if len(get) and self.EventCache: self.RestoreEvents(get)
        if check_event(self): pass
        else: return

        if check_tree(self): pass
        else: return

        path = self.Tree + "/" + self.EventName
        if check_number(self, path): pass
        else: return

        self.Files = None
        self.Files = self._get["get"]
        if not len(self.Files): return

        ev = EventGenerator()
        ev.Caller = "ANALYSIS::EVENT"
        ev.ImportSettings(self.ExportSettings())
        x = self.ShowLength
        if ev.EventStop is None: pass
        elif not len(x): pass
        else: ev.EventStop -= self.ShowLength[path]
        if ev.MakeEvents(self.SampleName, self): pass
        else: return

        if not self.EventCache: return
        self.DumpEvents()
        self.DumpTracer(self.SampleName)
        return True

    def __LoadSample__(self):
        tracer = self._CheckForTracer()

        l = len(self.SampleMap)
        if not l: self.SampleMap = ""

        for name in self.SampleMap:
            self.SampleName = name
            if not len(name): name = None
            if not tracer: pass
            else: self.RestoreTracer(tracer, name)
            search(self)
            self.__Event__()
            self.__Selection__()
            self.__FeatureAnalysis__()
            self.__Graph__()
        return True

    def Launch(self):
        self.__build__()
        self.__LoadSample__()
        self.__RandomSampler__()
        #self.__Optimizer__()
        #self.WhiteSpace()
        return True

    def preiteration(self) -> bool:
        self.GetAll = True
        x = self.makehashes()
        self.GetAll = False
        check_tree(self)

        check_event(self)
        if len(self.EventName): self.GetEvent = True
        if not self.EventName and not self.EventCache: pass
        else: self.RestoreEvents(sum(x["event"].values(), []))

        check_graph(self)
        if len(self.GraphName): self.GetGraph = True
        if not self.GraphName and not self.DataCache: pass
        else: self.RestoreGraphs(sum(x["graph"].values(), []))

        check_selection(self)
        if not len(self.SelectionName): pass
        else:
            self.RestoreSelections(sum(x["selection"].values(), []))
            self.GetSelection = True

        if self.triggered: return False
        else: self.Launch()

        return False



