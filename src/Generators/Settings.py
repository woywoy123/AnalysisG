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
            self.Name = Instance.__qualname__
        except AttributeError:
            try:
                self.Name = Instance.__name__
            except AttributeError:
                self.Name = type(Instance).__name__
        
        self.Module = Instance.__module__
        self.Path = self.Module + "." + self.Name
        self.Code = self.GetSourceFile(Instance)
    
    def CopyInstance(self, Instance):
        if callable(Instance):
            try:
                Inst = Instance()
            except:
                Inst = Instance
            Inst = Instance
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
        self.EventStart = 0
        self.EventStop = None

        self.ProjectName = "UNTITLED"
        self.SampleContainer = None


class _EventGenerator:

    def __init__(self):
        _General.__init__(self)

        self.Event = None
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

class _HDF5:

    def __init__(self):
        self._File = None
        self.Filename = "UNTITLED"
        self._ext = ".hdf5"
        self._iter = -1
        self._obj = {}
        self.VerboseLevel = 3
        self.Threads = 12
        self.chnk = 12

class _TrainingSample:

    def __init__(self):
        self.TrainingSampleName = False
        self.TrainingPercentage = 80

class _Optimization:
    def __init__(self):
        self.SplitSampleByNode = False
        self.kFolds = 10
        self.BatchSize = 10
        self.Model = None
        self.DebugMode = False
        self.ContinueTraining = False
        self.RunName = "UNTITLED"
        self.Epochs = 10
        self.Optimizer = None 
        self.Scheduler = None
        self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.VerbosityIncrement = 10

class _ModelEvaluator:
    
    def __init__(self):
        self._ModelDirectories = {}
        self._ModelSaves = {}
        self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.SampleNodes = {}
        self.Training = {}
        self.TestSample = {}
        self.PlotNodeStatistics = True 
        self.PlotTrainingStatistics = True
        self.PlotTrainSample = True 
        self.PlotTestSample = True 
        self.PlotEntireSample = True
        self.PlotEpochDebug = False

class _Analysis:

    def __init__(self):
        _General.__init__(self) 
        _EventGenerator.__init__(self)
        _GraphGenerator.__init__(self)
        _TrainingSample.__init__(self)
        _Optimization.__init__(self)
        _ModelEvaluator.__init__(self)
        self._SampleMap = {}
        
        self.EventCache = False
        self.DataCache = False

        self.DumpHDF5 = False
        self.DumpPickle = False
        self.OutputDirectory = "./"




class Settings(_General):
    
    def __init__(self):
        _General.__init__(self)
        
        if self.Caller == "EVENTGENERATOR":
            _EventGenerator.__init__(self)
        
        if self.Caller == "GRAPHGENERATOR":
            _GraphGenerator.__init__(self)

        if self.Caller == "HDF5":
            _HDF5.__init__(self)

        if self.Caller == "OPTIMIZATION":
            _Optimization.__init__(self)

        if self.Caller == "MODELEVALUATOR":
            _ModelEvaluator.__init__(self)

        if self.Caller == "ANALYSIS":
            _Analysis.__init__(self)

    def DumpSettings(self):
        return self.__dict__
    
    def RestoreSettings(self, inpt):
        for i in self.__dict__:
            if i not in inpt:
                continue
            if i == "_Code" or i == "SampleContainer" or i == "Caller":
               continue
            self.__dict__[i] = inpt[i]

    def AddCode(self, Instance):
        _c = _Code()
        _c.DumpCode(Instance)
        if _c not in self._Code:
            self._Code.append(_c)
        return self._Code[self._Code.index(_c)]

    def CopyInstance(self, Instance):
        _c = self.AddCode(Instance)
        return _c.CopyInstance(Instance) 
    
    def GetCode(self, inpt):
        for _c in inpt._Code:
            if _c in self._Code:
                continue
            self._Code.append(_c)
 
