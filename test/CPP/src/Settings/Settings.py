class _General:
    def __init__(self):
        self.Caller = "" if "Caller" not in self.__dict__ else self.Caller.upper()
        self.Verbose = 3
        self.Threads = 6
        self.chnk = 10
        self._Code = {}
        self.EventStart = -1
        self.EventStop = None
        self._Device = "cpu"

class _UpROOT:
    
    def __init__(self):
        self.StepSize = 1000
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.Files = {}

class _EventGenerator:
    
    def __init__(self):
        self.Event = None
        self.Files = {}

class _GraphGenerator:

    def __init__(self):
        self.EventGraph = None
        self.SelfLoop = True
        self.FullyConnect = True
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}

class _SelectionGenerator:
    
    def __init__(self):
        self.Selections = {}
        self.Merge = {}

class _Analysis:
    
    def __init__(self):
        _UpROOT.__init__(self)
        _Pickle.__init__(self)
        _EventGenerator.__init__(self) 
        _GraphGenerator.__init__(self) 
        _SelectionGenerator.__init__(self)
        self._cPWD = None
        self.ProjectName = "UNTITLED"
        self.OutputDirectory = None
        self.SampleMap = {}
        self.EventCache = False
        self.DataCache = False
        self.PurgeCache = False

class _Pickle:

    def __init__(self):
        self._ext = ".pkl"
        self.Verbose = 3

class Settings:
    
    def __init__(self, caller = False):
        if caller: self.Caller = caller
        _General.__init__(self)
        if self.Caller == "UP-ROOT": _UpROOT.__init__(self)
        if self.Caller == "PICKLER": _Pickle.__init__(self)
        if self.Caller == "EVENTGENERATOR": _EventGenerator.__init__(self) 
        if self.Caller == "GRAPHGENERATOR": _GraphGenerator.__init__(self) 
        if self.Caller == "SELECTIONGENERATOR": _SelectionGenerator.__init__(self)
        if self.Caller == "ANALYSIS": _Analysis.__init__(self)
    
    @property
    def Device(self):
        return self._Device

    @Device.setter
    def Device(self, val):
        import torch
        self._Device = val if val == "cuda" and torch.cuda.is_available() else "cpu"

    def ImportSettings(self, inpt):
        if not issubclass(type(inpt), Settings): return 
        s = Settings(self.Caller)
        for i in s.__dict__: setattr(self, i, getattr(inpt, i))
       
