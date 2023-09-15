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

def check_number(x, path):
    try: l = x.ShowLength[path]
    except KeyError: l = 0

    if x.EventStop is None: return True
    elif l == 0: return True
    elif l >= x.EventStop: return False
    return False

def search(x):
    f = {}
    f["SampleName"] = False
    f["Files"] = []
    f["get"] = {}

    x.GetAll = True
    f["SampleName"] = x.SampleName in x
    try: f["Files"] = x.SampleMap[x.SampleName]
    except KeyError: pass

    for i in f["Files"]:
        i = x.abs(i)
        if i in x: continue
        i = i.split("/")
        file, path = i[-1], "/".join(i[:-1])
        if path not in f["get"]: f["get"][path] = []
        f["get"][path].append(file)
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


    def __Graph__(self):
        if not check_graph(self): return
        if self.DataCache: self.RestoreGraphs()

        check_tree(self)
        path = self.Tree + "/" + self.GraphName
        if check_number(self, path): pass
        else: self.RestoreEvents()

        dic = {}
        if self.EventStop is not None: get1, get2 = {},{}
        else: get1, get2 = self._get["event"], self._get["graph"]

        for k in get1.values():
            dic.update({j : False for j in k})
        for k in get2.values():
            try:
                for j in k: del dic[j]
            except KeyError: pass
        if len(dic): self.RestoreEvents(list(dic))
        else: return

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
        get = []
        self.GetAll = True
        for i in self._get["event"]:
            if not self.EventCache: break
            get += self._get["event"][i]
        if len(get): self.RestoreEvents(get)
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
        if ev.MakeEvents(self.SampleName, self): pass
        else: return False

        if not self.EventCache: return True
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
            self.__FeatureAnalysis__()
            self.__Graph__()
        return True

    def Launch(self):
        self.__build__()
        self.__LoadSample__()
        #self.__Selection__()
        self.__RandomSampler__()
        #self.__Optimizer__()
        #self.WhiteSpace()
        return True

    def preiteration(self) -> bool:
        if self.triggered: pass
        else: self.Launch()
        x = self.makehashes()
        if len(x["event"]): pass
        elif len(x["graph"]): pass
        elif len(x["selection"]): pass
        elif not len(self._get): pass
        else: return self.EmptySampleList()

        if not len(self.SelectionName):
            try: self.SelectionName = self.ShowSelections[0]
            except IndexError: pass
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: pass
        return False



