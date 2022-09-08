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

    def Analysis(self):
        self.GenerateDataLoader()
        self.EventGenerator() 
        self.Computations()
        self.Notification()

        self.EventCache = False
        self.DataCache = False

        self.EventStart = 0
        self.EventEnd = None
        
        self.ProjectName = "UNTITLED" 
        self.OutputDirectory = None
        
        self.Tree = "nominal"
        self.SelfLoop = True 
        self.FullyConnect = True

        self.DumpHDF5 = True
        
        
