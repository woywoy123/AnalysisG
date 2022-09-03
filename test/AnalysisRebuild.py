from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.IO import WriteDirectory, Directories
from AnalysisTopGNN.Generators import EventGenerator
import os

class AnalysisNew(WriteDirectory, Directories, Notification):

    def __init__(self):
        Notification.__init__(self)
        super().__init__()
        
        # ==== EventGenerator Variables ==== #
        self.Event = None 

        # ==== Public Variables ==== #
        self.Caller = "Analysis"
        self.VerboseLevel = 2
        self.ProjectName = "UNTITLED" 
        self.OutputDirectory = None

        # ==== EventGenerator Public Variables ==== #
        self.EventCache = False
        self.EventStart = 0
        self.EventEnd = -1
        self.CPUThreads = 12
        self.Tree = "nominal"

        # ==== Hidden Internal Variables ==== #
        self._SampleMap = {}

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

        if self.EventCache:
            if self.Event == None:
                self.Fail("NO EVENT IMPLEMENTATION GIVEN. EXITING...")
            Compiler = EventGenerator(None, self.Verbose, self.EventStart, self.EventEnd)
            Compiler.EventImplementation = self.Event
            Compiler.VerboseLevel = self.VerboseLevel
            Compiler.Threads = self.CPUThreads
        
        for i in InputMap:
            F = InputMap[i]
            if self.EventCache:
                F = F.split("/")
                BaseDir = "/".join(F[:-1])
                if BaseDir not in Compiler.Files:
                    Compiler.Files[BaseDir] = []
                Compiler.Files[BaseDir] += [F[-1]]
        
        if self.EventCache:
            Compiler.SpawnEvents()
            Single = False
            if self.CPUThreads == 1:
                Single = True
            Compiler.CompileEvent(Single)

    def Launch(self):
        self.__BuildRootStructure()
        EventMap = self.__CheckFiles(self._SampleMap)
        EventMap = self.__BuildSampleDirectory(EventMap, "EventCache")
        self.__BuildCache(EventMap) 



