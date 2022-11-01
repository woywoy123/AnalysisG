from AnalysisTopGNN.Tools import Tools
import torch 

class _Code(Tools):

    def __init__(self):
        self.Name = None 
        self.Module = None 
        self.Path = None 
        self.Code = None 

    def DumpCode(self, Instance):
        try:
            self.Name = Instance.__name__ 
        except AttributeError:
            self.Name = type(Instance).__name__

        self.Module = Instance.__module__
        self.Path = self.Module + "." + self.Name
        self.Code = self.GetSourceFile(Instance)
    
    def CopyInstance(self, Instance):
        if callable(Instance):
            Instance = Instance()

        if self.Name == None:
            self.DumpCode(Instance)
            
        _, inst = self.GetObjectFromString(self.Module, self.Name)
        return inst

    def __eq__(self, other):
        for i in other.__dict__:
            if self.__dict__[i] == other.__dict__[i]:
                continue 
            return False
        return True

class _General:

    def __init__(self):
        self.VerboseLevel = 3
        self.chnk = 12
        self.Threads = 12
        self.Tree = None
        self._dump = False
        self._Code = []

        self.ProjectName = "UNTITLED"


class _EventGenerator:

    def __init__(self):
        _General.__init__(self)
        self.Tracer = None 

        self.Event = None
        self.EventStart = 0
        self.EventStop = None
        self.InputDirectory = {}

class _GraphGenerator:
    
    def __init__(self):
        _General.__init__(self)
        self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.SelfLoop = True 
        self.FullyConnect = True 
        self.EventGraph = None
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}

class Settings(_General):
    
    def __init__(self):
        _General.__init__(self)
        
        if self.Caller == "EVENTGENERATOR":
            _EventGenerator.__init__(self)
        if self.Caller == "GRAPHGENERATOR":
            _GraphGenerator.__init__(self)
    
    def DumpSettings(self):
        pass

    def RestoreSettings(self, inpt):
        pass
    
    def AddCode(self, Instance):
        _c = _Code()
        _c.DumpCode(Instance)
        if _c not in self._Code:
            self._Code.append(_c)
        return self._Code[self._Code.index(_c)]

    def CopyInstance(self, Instance):
        _c = self.AddCode(Instance)
        return _c.CopyInstance(Instance) 






class _Settings:

    def __init__(self):
        self.EventCache = False
        self.DataCache = False

        self.VerboseLevel = 3
        self.Threads = 12
        self.chnk = 12
        
        self.EventStart = 0
        self.EventStop = None 

        self.DumpHDF5 = False
        self.DumpPickle = False

        self.OutputDirectory = False

        self.ProjectName = "UNTITLED"
        self.Tracer = None
        self.Tree = False
        self._PullCode = False
        self.Device = "cpu"

        self.InputDirectory = {}
        
        # ==== Optimization ==== #
        self.Model = None
        self.Epochs = 10
        self.kFolds = 10

        self.Scheduler = None
        self.Optimizer = None
        self.BatchSize = 20 
        
        self.Tree = None
        self.SplitSampleByNode = False
        self.ContinueTraining = False
        self.RunName = None 
        self.DebugMode = False
         
    def DumpSettings(self, other):
        
        Unique = {}
        for i in self.__dict__:
            if i not in other.__dict__:
                continue
            if self.__dict__[i] == other.__dict__[i]:
                continue

            Unique[i] = other.__dict__[i]
        return Unique

    def ImportSettings(self, inpt):
        
        for i in inpt:
            self.__dict__[i] = inpt[i]
