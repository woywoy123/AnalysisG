from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.IO import WriteDirectory, Directories, PickleObject, UnpickleObject
from AnalysisTopGNN.Generators import EventGenerator, GenerateDataLoader
from AnalysisTopGNN.Parameters import Parameters
import os
import random

class FeatureAnalysis(Notification, Parameters):
    def __init__(self):
        Notification.__init__(self)
        self.GenerateDataLoader()

    def FeatureAnalysis(self, Input, Fx_n, Fx, Fx_m):
        Input = self.EventGraph(Input)
        try: 
            if "GraphFeatures" in Fx_m:
                Fx(Input.Event)
            if "NodeFeatures" in Fx_m:
                for i in Input.Particles:
                    Fx(i)
            if "EdgeFeatures" in Fx_m:
                for i in Input.Particles:
                    for j in Input.Particles:
                        Fx(i, j)
            return []
    
        except AttributeError:
            fail = str(sys.exc_info()[1]).replace("'", "").split(" ")
            return ["FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + " ".join(fail)]
    
    def TestObjectAttributes(self, Event):
        samples = [Event[k][self.Tree] for k in random.sample(list(Event), 10)]
        Failed = []
        Features = []
        Features += list(self.GraphAttribute)
        Features += list(self.NodeAttribute)
        Features += list(self.EdgeAttribute)
        for i in samples:
            # Test the Graph Attributes 
            for gn, gfx in self.GraphAttribute.items():
                Failed += self.FeatureAnalysis(i, gn, gfx, "GraphFeatures")
            for nn, nfx in self.NodeAttribute.items():
                Failed += self.FeatureAnalysis(i, nn, nfx, "NodeFeatures")
            for en, efx in self.EdgeAttribute.items():
                Failed += self.FeatureAnalysis(i, en, efx, "EdgeFeatures")
        if len(Failed) == 0:
            return 
    
        self.Warning("------------- Feature Errors -------------------")
        for i in list(set(Failed)):
            self.Warning(i)
    
        self.Warning("------------------------------------------------")
        if len(list(set(Failed))) == int(len(Features)):
            self.Fail("NONE OF THE FEATURES PROVIDED WERE SUCCESSFUL!")
 
class AnalysisNew(WriteDirectory, GenerateDataLoader, Directories, FeatureAnalysis):

    def __init__(self):
        Notification.__init__(self)
        super().__init__()
        
        self.Analysis()
        self.Caller = "Analysis"

        # ==== Hidden Internal Variables ==== #
        self._SampleMap = {}
        self._iter = 0

    def InputSample(self, Name, Directory):
        
        self.Notify("=====================================")
        if isinstance(Directory, str):
            Directory = [Directory] 
        
        if Name not in self._SampleMap:
            self.Notify("Added Sample: " + Name)
            self._SampleMap[Name] = Directory
        else:
            self.Warning(Name + " already exists! Checking if Directory already exists in List.")
            for i in Directory: 
                if i not in self._SampleMap[Name]:
                    self._SampleMap[Name].append(i)
                else:
                    self.Warning(i + " already exists! Skipping...")
        self.Notify("=====================================")

    def __BuildRootStructure(self): 

        # ==== Builds the Main Directory ==== #
        if self.OutputDirectory == None:
            self.OutputDirectory = self.pwd
        if self.OutputDirectory.endswith("/"):
            self.OutputDirectory = self.OutputDirectory[:-1]

        self.Notify("BUILDING PROJECT " + self.ProjectName + " UNDER: " + self.OutputDirectory)
        self.MakeDir(self.OutputDirectory + "/" + self.ProjectName)
        self.ChangeDirToRoot(self.OutputDirectory + "/" + self.ProjectName)
        
        self.MakeDir("./EventCache")
        self.MakeDir("./DataCache")

    def __CheckFiles(self, InputMap):
        OutputMap = {}
        for i in InputMap:
            OutputMap[i] = [t for k in InputMap[i] for t in self.ListFilesInDir(k)]
        return OutputMap 

    def __BuildSampleDirectory(self, InputMap, CacheType):
        OutputMap = {}
        for i in InputMap:
            self.MakeDir("./" + CacheType + "/" + i)
            for j in InputMap[i]:
                basename = j.split("/")[-1].split(".")[0]
                self.MakeDir("./" + CacheType + "/" + i + "/" + basename)
                OutputMap["./" + CacheType + "/" + i + "/" + basename] = j
        return OutputMap

    def __BuildCache(self, InputMap):
        def EventCache(BaseName, FileName):
            if self.Event == None:
                self.Fail("NO EVENT IMPLEMENTATION GIVEN. EXITING...")
            Compiler = EventGenerator(None, self.Verbose, self.EventStart, self.EventEnd)
            Compiler.Event = self.Event
            Compiler.VerboseLevel = self.VerboseLevel
            Compiler.Threads = self.CPUThreads
            Compiler.Files[BaseName] = [FileName]
            Compiler.SpawnEvents()
            Compiler.CompileEvent()
            return Compiler
         
        def DataCache(EventInstance):
            if self.EventGraph == None:
                self.Fail("NO EVENTGRAPH IMPLEMENTATION GIVEN. EXITING...")
            self.TestObjectAttributes(EventInstance.Events)
            self.AddSample(EventInstance, self.Tree, self.SelfLoop, self.FullyConnect)


        for i in InputMap:
            F = InputMap[i].split("/")
            BaseDir = "/".join(F[:-1])
            FileName = F[-1] 
            
            if self.EventCache and FileName.endswith(".root"):
                ev = EventCache(BaseDir, FileName) 
                PickleObject(ev, FileName.replace(".root", ""), i) 

            if self.DataCache and FileName.endswith(".pkl"):
                ev = UnpickleObject(BaseDir + "/" + FileName)
                DataCache(ev)

    def Launch(self):
        self.__BuildRootStructure()
        
        if self.EventCache:
            self.Notify("------ Checking for ROOT Files -------")
            EventMap = self.__CheckFiles(self._SampleMap)
            EventMap = self.__BuildSampleDirectory(EventMap, "EventCache")
            self.__BuildCache(EventMap) 
        
        if self.DataCache:
            self.Notify("------ Checking for EventCached Files -------")          
            Cache = self.__CheckFiles({"" : ["./EventCache"]})
            DirMap = {i.split("/")[2] : [] for i in Cache[""]}
            for i in Cache[""]:
                DirMap[i.split("/")[2]].append(i)
            self.__BuildSampleDirectory(DirMap, "DataCache")
            
            DataMap = { "/".join(i.split("/")[:-1]).replace("./EventCache", "./DataCache") : i for i in Cache[""]}
            self.__BuildCache(DataMap)
            self.ProcessSamples()
        

