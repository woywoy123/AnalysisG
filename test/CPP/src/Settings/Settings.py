class _General:
    def __init__(self):
        self.Caller = "" if "Caller" not in self.__dict__ else self.Caller.upper()
        self.Verbose = 3
        self.Threads = 6
        self.chnk = 10
        self._Code = {}
        self.EventStart = -1
        self.EventStop = None

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
        self.Device = "cpu"
        self.EventGraph = None
        self.SelfLoop = True
        self.FullyConnect = True
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}

class _Pickle:

    def __init__(self):
        self._ext = ".pkl"
        self.Verbose = 3

class Settings:
    
    def __init__(self):
        _General.__init__(self)
        if self.Caller == "UP-ROOT": _UpROOT.__init__(self)
        if self.Caller == "PICKLER": _Pickle.__init__(self)
        if self.Caller == "EVENTGENERATOR": _EventGenerator.__init__(self) 
        if self.Caller == "GRAPHGENERATOR": _GraphGenerator.__init__(self) 
