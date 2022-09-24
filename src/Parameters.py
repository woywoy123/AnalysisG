class Parameters:

    def __init__(self):
        pass
    
    def Notification(self):
        self.Verbose = True 
        self.VerboseLevel = 1

    def Computations(self):
        self.Threads = 12
        self.Device = "cpu"

    def EventGenerator(self):
        self.Events = {}
        self.Files = {}
        self.Event = None
        
        self.FileEventIndex = {}
        self.FileEventIndex["Start"] = []
        self.FileEventIndex["End"] = []
        self.chnk = 100
    
    def GenerateDataLoader(self):
        self.NEvents = -1
        self.CleanUp = True 
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}

        self.DataContainer = {}
        self.TrainingSample = {}
        self.ValidationSample = {}
        
        self.FileTraces = {i : [] for i in ["Tree", "Start", "End", "Level", "SelfLoop", "Samples"]}

        self.EventGraph = None
        self.ValidationSize = 50
        self.chnk = 10

        
    def Optimizer(self):
        self.LearningRate = 0.0001
        self.WeightDecay = 0.001
        self.kFold = 10
        self.Epochs = 10
        self.BatchSize = 10
        self.StartEpoch = 0
        self.Model = None
        self.Scheduler = None
        self.RunName = "UNTITLED"
        self.RunDir = "_Models"
        self.DefaultOptimizer = "ADAM"
        self.DefaultScheduler = "ExponentialR"
        self.SchedulerParams = {"gamma" : 0.9}

        self.ONNX_Export = False
        self.TorchScript_Export = True
        self.Debug = False
 
        self.Training = True
        self.T_Features = {}
        self.CacheDir = None

    def Analysis(self):
        self.GenerateDataLoader()
        self.EventGenerator() 
        self.Computations()
        self.Optimizer()
        self.Notification()

        self.EventCache = False
        self.DataCache = False
        self.chnk = 10

        self.EventStart = 0
        self.EventEnd = None
        
        self.ProjectName = "UNTITLED" 
        self.OutputDirectory = None
        
        self.Tree = "nominal"
        self.SelfLoop = True 
        self.FullyConnect = True

        self.DumpHDF5 = True
        self.MergeSamples = False

        self.DataCacheDir = "./HDF5"
        self.GenerateTrainingSample = False
