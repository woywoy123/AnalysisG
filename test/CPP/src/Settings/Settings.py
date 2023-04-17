class _General:
    def __init__(self):
        self.Caller = "" if "Caller" not in self.__dict__ else self.Caller.upper()
        self.Verbose = 3
        self.Threads = 6
        self.chnk = 10
        self._Code = {}

class _UpROOT:
    
    def __init__(self):
        self.StepSize = 100000
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.ROOTFile = None

class _EventGenerator:
    
    def __init__(self):
        self.EventStart = 0
        self.EventStop = -1
        self.Event = None
        self.Files = {}

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
