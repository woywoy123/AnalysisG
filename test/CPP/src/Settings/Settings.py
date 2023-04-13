class _General:
    def __init__(self):
        self.Verbose = 3
        self.Caller = "" if "Caller" not in self.__dict__ else self.Caller.upper()

class _UpROOT:
    
    def __init__(self):
        self.StepSize = 5000
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.ROOTFile = None

class Settings:
    
    def __init__(self):
        _General.__init__(self)
        if self.Caller == "UP-ROOT": _UpROOT.__init__(self)
        
