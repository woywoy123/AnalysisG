from Functions.Tools.Alerting import Notification 
from Functions.IO.Files import WriteDirectory, Directories 
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.EventGenerator import EventGenerator 
from Functions.Event.DataLoader import GenerateDataLoader
import os 
import sys
import random
import shutil  

class Unification(WriteDirectory, Directories, GenerateDataLoader, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        GenerateDataLoader.__init__(self)
        Notification.__init__(self)
        
        # ==== Initial variables ==== #
        self.Caller = "Unification"
        self.VerboseLevel = 2
        self.Verbose = True 
        self.ProjectName = "UNTITLED"
        self.OutputDir = None
        
        # ==== Hidden internal variables ==== #
        self._MainDirectory = None
        self._SampleDir = {}
        self._EventGeneratorDir = "EventGenerator"
        self._DataLoaderDir = "DataLoader"

        # ==== Event Generator variables ==== # 
        self.EventImplementation = None  
        self.CompileSingleThread = False
        self.CPUThreads = 12
        self.NEvent_Start = 0
        self.NEvent_Stop = -1
        self.EventCache = False
       
        # ===== DataLoader variables ===== #
        self.DataCache = False
        self.DataCacheInput = []
        self.Tree = "nominal"
        self.EventGraph = None
        self.ValidationSampleSize = 50
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}
        self.Device_S = "cpu" 
        self.SelfLoop = True
        self.FullyConnect = True

    def __ClearTempVariable(self):
        self._TMP = None
        self._CommonDir = None


    def __CheckSettings(self):
        if self.ProjectName == "UNTITLED":
            self.Warning("NAME VARIABLE IS SET TO: " + self.ProjectName)
        
        if self.OutputDir == None:
            self.Warning("NO OUTPUT DIRECTORY DEFINED. USING CURRENT DIRECTORY: \n" + self.pwd)
            self.OutputDir = self.pwd

        if len(self._SampleDir) == 0 and self.EventCache:
            self.Fail("NO SAMPLES GIVEN. EXITING...")
        elif self.DataCache and self.EventGraph == None:
            self.Fail("NO EVENT GRAPH HAS BEEN GIVEN. EXITING...")
        else:
            self._TMP = {}
            for k, l in self._SampleDir.items():
                self._TMP[k] = self.__CheckFiles(l)
        
        if self.EventCache == True and self.EventImplementation == None:
            from Functions.Event.Implementations.Event import Event
            self.Warning("No 'EventImplementation' provided. Assuming 'Functions.Event.Implementations.Event'.")
            self.EventImplementation = Event()

        if self.DataCache == True:
            attrs = 0
            if len(list(self.EdgeAttribute)) == 0:
                self.Warning("NO EDGE FEATURES PROVIDED")
                attrs+=1
            if len(list(self.NodeAttribute)) == 0:
                self.Warning("NO NODE FEATURES PROVIDED")
                attrs+=1
            if len(list(self.GraphAttribute)) == 0:
                self.Warning("NO GRAPH FEATURES PROVIDED")
                attrs+=1
            if attrs == 3:
                self.Fail("NO ATTRIBUTES DEFINED!")
    
    def __BuildStructure(self):
        # Initialize the ROOT directory
        if self.OutputDir.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]
        if self.ProjectName.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]

        self.ChangeDirToRoot(self.OutputDir)
        self.MakeDir(self.ProjectName)
        self._MainDirectory = self.OutputDir + "/" + self.ProjectName
        self.ChangeDirToRoot(self._MainDirectory)

        # Make a directory for the EventGenerator files 
        if self.EventCache == True:
            self._CommonDir = os.path.commonpath([i for name in self._TMP for i in list(self._TMP[name])])
            for name in self._SampleDir:
                for FileDir in self._SampleDir[name]:
                    self.MakeDir(self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, ""))
        
        if self.DataCache == True:
            self.MakeDir(self._DataLoaderDir)
            self.MakeDir(self._DataLoaderDir + "Test")
            self.MakeDir(self._DataLoaderDir + "FileTraceDump")

    
    def __CheckFiles(self, Directory):
        out = {}
        for i in Directory:
            val = self.ListFilesInDir(i)
            for p in val: 
                f, key  = p.split("/")[-1], "/".join(p.split("/")[:-1])
                if key not in out:
                    out[key] = []
                if f not in out[key]:
                    out[key].append(f)
        return out
    
    def __StartEventGenerator(self, FileDir, Samples, OutDir): 
        for S in Samples:
            ev = EventGenerator(None, self.Verbose, self.NEvent_Start, self.NEvent_Stop)
            ev.EventImplementation = self.EventImplementation
            ev.Files = {FileDir : [S]}
            ev.VerboseLevel = 0
            ev.Threads = self.CPUThreads 
            ev.SpawnEvents()
            ev.CompileEvent(self.CompileSingleThread)
            PickleObject(ev, S.replace(".root", ""), OutDir)
    
   
    def InputSample(self, Name,  Directory):
        if isinstance(Name, str) == False:
            self.Warning("NAME NOT A STRING!")
            return 
        if Name not in self._SampleDir:
            self._SampleDir[Name] = []

        if isinstance(Directory, list):
            self._SampleDir[Name] += Directory 

        elif isinstance(Directory, str):
            self._SampleDir[Name].append(Directory)

        else:
            self.Warning("INPUT DIRECTORY NOT VALID (STRING OR LIST)!")
            return 

    def __FeatureAnalysis(self, Input, Fx_n, Fx, Fx_m):

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
            return ["FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + fail[-1]]

    def __TestObjectAttributes(self, Event):
        samples = [Event[k][self.Tree] for k in random.sample(list(Event), 10)]
        Failed = []
        Features = []
        Features += list(self.GraphAttribute)
        Features += list(self.NodeAttribute)
        Features += list(self.EdgeAttribute)
        for i in samples:
            # Test the Graph Attributes 
            for gn, gfx in self.GraphAttribute.items():
                Failed += self.__FeatureAnalysis(i, gn, gfx, "GraphFeatures")
            for nn, nfx in self.NodeAttribute.items():
                Failed += self.__FeatureAnalysis(i, nn, nfx, "NodeFeatures")
            for en, efx in self.EdgeAttribute.items():
                Failed += self.__FeatureAnalysis(i, en, efx, "EdgeFeatures")
        if len(Failed) == 0:
            return 
        self.Warning("------------- Feature Errors -------------------")
        for i in list(set(Failed)):
            self.Warning(i)
        self.Warning("------------------------------------------------")
        if len(list(set(Failed))) == int(len(Features)):
            self.Fail("NONE OF THE FEATURES PROVIDED WERE SUCCESSFUL!")

    def Launch(self):
        self.__CheckSettings()
        self.__BuildStructure()

        if self.EventCache:
            for name in self._TMP:
                for FileDir, Samples in self._TMP[name].items():
                    self.Notify("STARTING THE COMPILATION OF NEW DIRECTORY: " + FileDir)
                    FileOut = self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, "")
                    self.__StartEventGenerator(FileDir, Samples, FileOut)

        if self.DataCache:
            shutil.rmtree(self._DataLoaderDir)
            shutil.rmtree(self._DataLoaderDir + "FileTraceDump")
            shutil.rmtree(self._DataLoaderDir + "Test")
            self.__BuildStructure()
            Samples = self.__CheckFiles([self._EventGeneratorDir])
            for Dir, Samples in Samples.items():
                Switch = True
                for S in Samples: 
                    ev = UnpickleObject(S, Dir)
                    if Switch:
                        self.__TestObjectAttributes(ev.Events)
                        Switch = False
                    self.AddSample(ev, self.Tree, self.SelfLoop, self.FullyConnect, -1)
                    break
            PickleObject(self.FileTraces, "FileTraces", self._DataLoaderDir + "FileTraceDump")
            self.MakeTrainingSample(self.ValidationSampleSize) 

            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            for i in self.DataContainer:
                address = hex(id(self.DataContainer[i]))
                Exp.ExportEventGraph(self.DataContainer[i], str(address), self._DataLoaderDir)
                self.Notify("!!DUMPED EVENT " + str(int(self.DataContainer[i].i)) + "/" + str(len(self.DataContainer)))
            
            for i in self.TrainingSample:
                self.TrainingSample[i] = [str(hex(id(k))) for k in self.TrainingSample[i]]
            for i in self.ValidationSample:
                self.ValidationSample[i] = [str(hex(id(k))) for k in self.ValidationSample[i]]
            
            PickleObject(self.TrainingSample, "TrainingSample", self._DataLoaderDir + "FileTraceDump")
            PickleObject(self.ValidationSample, "ValidationSample", self._DataLoaderDir + "FileTraceDump")
        
        else:
            self.TrainingSample = UnpickleObject("TrainingSample", self._DataLoaderDir + "FileTraceDump")
            self.ValidationSample = UnpickleObject("ValidationSample", self._DataLoaderDir + "FileTraceDump")
            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            for i in self.TrainingSample:
                for j in range(len(self.TrainingSample[i])):
                    self.TrainingSample[i][j] = Exp.ImportEventGraph(self.TrainingSample[i][j], self._DataLoaderDir)








