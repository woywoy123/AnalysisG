from training.methods import *
from training.config import *

class DefaultCfg:

    def __init__(self, cks = None): 
        self.cks = cks
        self._name = ""

    def check(self):
        if self.cks is None: return 
        for i in self.cks: assert i(self)

    @property
    def name(self): return self._name
    @name.setter
    def name(self, val): self._name = val.lower()

class BaseCfg:

    def __init__(self):
        self.OutputPath       = "./ProjectName"
        self.Threads          = 4

        self.IntraThreads     = None
        self.DebugMode        = False

class GraphCfg(DefaultCfg):

    def __init__(self, graph_name = ""):
        DefaultCfg.__init__(self, [graph_method])

        self.name  = graph_name
        self.BuildCache = None
        self.GraphCache = None 

    def Graph(self):
        self.check()
        gr = graph_method(self, True)
        if gr is None: return None
        return gr() 

class EventCfg(DefaultCfg):

    def __init__(self, event_name = ""):
        DefaultCfg.__init__(self, [event_method])
        self.name  = event_name

    def Event(self):
        self.check()
        ev = event_method(self, True)
        if ev is None: return None
        return ev()

class CosmeticCfg(DefaultCfg):

    def __init__(self):
        DefaultCfg.__init__(self)

        self.nBins            = None
        self.MaxRange         = None
        self.SetLogY          = None

        self.VarPt            = None #"var-pt"
        self.VarEta           = None #"var-eta"
        self.VarPhi           = None #"var-phi"
        self.VarEnergy        = None #"var-energy"
        self.Targets          = None #["top_edge"]


class SampleCfg(DefaultCfg):

    def __init__(self):
        DefaultCfg.__init__(self)
        self.event = EventCfg()
        self.graph = GraphCfg()
        self._path  = None
        self._dset  = None
        self.samples = []

    def fetch_path(self, val): 
        self._path = val
        dsets = dataset_method(val)
        for d in dsets:
            for f in dsets[d]:
                cg = SampleCfg()
                cg.event = self.event
                cg.graph = self.graph
                cg._path  = f
                cg._dset  = d
                self.samples.append(cg)

    def __iter__(self): 
        self.it = iter(self.samples)
        return self

    def __next__(self): return next(self.it)


