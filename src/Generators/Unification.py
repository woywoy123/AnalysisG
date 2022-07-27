from AnalysisTopGNN.Tools import Notification 
from AnalysisTopGNN.IO import WriteDirectory, Directories, PickleObject, UnpickleObject, ExportToDataScience
from AnalysisTopGNN.Generators import EventGenerator, GenerateDataLoader, Optimizer
from AnalysisTopGNN.Events import Event
import os 
import sys
import random
import shutil  
import torch

class Analysis(Optimizer, WriteDirectory, Directories, GenerateDataLoader, Notification):
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
        self._launchDir = os.getcwd()

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
        self.Device = "cpu" 
        self.SelfLoop = True
        self.FullyConnect = True

        # ===== Optimizer variables ====== # 
        self.LearningRate = 0.0001
        self.WeightDecay = 0.001
        self.kFold = 4
        self.Epochs = 0
        self.BatchSize = 10
        self.Model = None
        self.RunName = "TESTMODEL"
        self.DefaultOptimizer = "ADAM"
        self.ONNX_Export = False
        self.TorchScript_Export = False
        self.Training = True
        self.Debug = False

        # ===== Constant variables ====== #
        self.RunDir = None
        self.T_Features = {}


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

    def __CheckSettings(self, Optimizer = False):
        if Optimizer == True and self.Model != None:
            self.FileTraces = UnpickleObject("FileTraces", self._DataLoaderDir + "FileTraceDump")
            self.TrainingSample = UnpickleObject("TrainingSample", self._DataLoaderDir + "FileTraceDump")
            self.ValidationSample = UnpickleObject("ValidationSample", self._DataLoaderDir + "FileTraceDump")
            tmp = self.VerboseLevel 
            self.VerboseLevel = 0 
            events = self.__CheckFiles(self._DataLoaderDir)["DataLoader"]
            self.VerboseLevel = tmp
            self.Notify("FOUND " + str(len(events)) + " EVENTS IN THE DATALOADER DIRECTORY.")
            self.Notify("CHECKING IF EVENTS ARE CONSISTENT.")
            counter = 0
            for i in [p for n in self.TrainingSample for p in self.TrainingSample[n]]:
                if str(i + ".hdf5") not in events:
                    self.Fail("MISSING FILE -> " + i + ".hdf5 FROM TRAINING SAMPLE! EXITING...")
                counter += 1
            for i in [p for n in self.ValidationSample for p in self.ValidationSample[n]]:
                if str(i + ".hdf5") not in events:
                    self.Fail("MISSING FILE -> " + i + ".hdf5 FROM VALIDATION SAMPLE! EXITING...")
                counter += 1
            self.Notify("SUCCESSFULLY VERIFIED " + str(counter) + " FILES.")
            self.CacheDir = self._DataLoaderDir
            
            tmp = {}
            if 0 in self.TrainingSample:
                self.Warning("ZERO NODE EVENTS DETECTED IN SAMPLES. REMOVING THEM.")
                self.TrainingSample.pop(0) 

            if 0 in self.ValidationSample:
                self.ValidationSample.pop(0) 
            return 

        if self.ProjectName == "UNTITLED":
            self.Warning("NAME VARIABLE IS SET TO: " + self.ProjectName)
        
        if self.OutputDir == None:
            self.Warning("NO OUTPUT DIRECTORY DEFINED. USING CURRENT DIRECTORY: \n" + self.pwd)
            self.OutputDir = self.pwd
        self.Notify("BUILDING THE PROJECT: '" + self.ProjectName + "' IN " + self.OutputDir)
        self._MainDirectory = self.OutputDir 
        if self.ProjectName != self.OutputDir.split("/")[-1]:
            self._MainDirectory += "/" + self.ProjectName
        
        for k, l in self._SampleDir.items():
            self._SampleDir[k] = self.__CheckFiles(l)
            if len(self._SampleDir[k]) == 0:
                self._SampleDir.pop(k, None)
 
        # Case 1: We want to only create the event generator caches 
        # - Check if a sample has been provided and that they resolve to root files       
        if self.EventCache == True and len(self._SampleDir) == 0:
            self.Fail("NO VALID SAMPLES GIVEN/FOUND. EXITING...")
        # - Revert to nominal implementation if no event implementation has been given 
        if self.EventCache == True and self.EventImplementation == None:
            self.Warning("No 'EventImplementation' provided. Assuming 'Functions.Event.Implementations.Event'.")
            self.EventImplementation = Event()
        
        # Case 2: We want to create the dataloader from the event generator cached/just created
        # - Check if we can find a cached event generator instance 
        cached = self.__CheckFiles(self._MainDirectory + "/" + self._EventGeneratorDir)
        if len(cached) == 0 and self.DataCache == True and len(self._SampleDir) == 0:
            self.Fail("NOTHING CACHED OR GIVEN SAMPLES NOT FOUND. EXITING...") 
        
        if self.DataCache == True and self.EventGraph == None:
            self.Fail("CAN'T CREATE SAMPLE EVENT GRAPH FOR DATALOADER. NO EVENTGRAPH FOUND. See 'Functions.Event.Implementations.EventGraph'")
        
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
                self.Fail("NO FEATURES DEFINED! EXITING...")

        if self.DataCache == True and self.EventCache == False:
            self._SampleDir[""] = {}
            self._CommonDir = self._EventGeneratorDir
            for Dir, Files in cached.items():
                self._SampleDir[""][Dir] = Files

    def __BuildStructure(self):
        # Initialize the ROOT directory
        if self.OutputDir.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]
        if self.ProjectName.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]
       
        self.MakeDir(self._MainDirectory)
        self.ChangeDirToRoot(self._MainDirectory)

        if self.EventCache == True:
            shutil.rmtree(self._EventGeneratorDir, ignore_errors = True)
            self._CommonDir = os.path.commonpath([i for name in self._SampleDir for i in list(self._SampleDir[name])])
            for name in self._SampleDir:
                for FileDir in self._SampleDir[name]:
                    self.MakeDir(self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, ""))
        
        if self.DataCache == True:
            shutil.rmtree(self._DataLoaderDir, ignore_errors = True)
            shutil.rmtree(self._DataLoaderDir + "FileTraceDump", ignore_errors = True)
            shutil.rmtree(self._DataLoaderDir + "Test", ignore_errors = True)
            
            self.MakeDir(self._DataLoaderDir)
            self.MakeDir(self._DataLoaderDir + "Test")
            self.MakeDir(self._DataLoaderDir + "FileTraceDump")
    
    def __CheckFiles(self, Directory):
        if isinstance(Directory, str):
            Directory = [Directory]
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
    
    def __BuildSampleCache(self): 
        def __FeatureAnalysis(Input, Fx_n, Fx, Fx_m):
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

        def __TestObjectAttributes(Event):
            samples = [Event[k][self.Tree] for k in random.sample(list(Event), 10)]
            Failed = []
            Features = []
            Features += list(self.GraphAttribute)
            Features += list(self.NodeAttribute)
            Features += list(self.EdgeAttribute)
            for i in samples:
                # Test the Graph Attributes 
                for gn, gfx in self.GraphAttribute.items():
                    Failed += __FeatureAnalysis(i, gn, gfx, "GraphFeatures")
                for nn, nfx in self.NodeAttribute.items():
                    Failed += __FeatureAnalysis(i, nn, nfx, "NodeFeatures")
                for en, efx in self.EdgeAttribute.items():
                    Failed += __FeatureAnalysis(i, en, efx, "EdgeFeatures")
            if len(Failed) == 0:
                return 
            self.Warning("------------- Feature Errors -------------------")
            for i in list(set(Failed)):
                self.Warning(i)
            self.Warning("------------------------------------------------")
            if len(list(set(Failed))) == int(len(Features)):
                self.Fail("NONE OF THE FEATURES PROVIDED WERE SUCCESSFUL!")
        
        for name in self._SampleDir:
            for FileDir, Samples in self._SampleDir[name].items():
                self.Notify("STARTING THE COMPILATION OF NEW DIRECTORY: " + FileDir)
                FileOut = self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, "")
                CheckSample = True
                for S in Samples:
                    if self.EventImplementation != None:
                        ev = EventGenerator(None, self.Verbose, self.NEvent_Start, self.NEvent_Stop)
                        ev.EventImplementation = self.EventImplementation
                        ev.Files = {FileDir : [S]}
                        ev.VerboseLevel = 0
                        ev.Threads = self.CPUThreads 
                        ev.SpawnEvents()
                        ev.CompileEvent(self.CompileSingleThread)
                    else:
                        ev = UnpickleObject(S, FileDir)
                  
                    if CheckSample:
                        __TestObjectAttributes(ev.Events)
                        CheckSample = False
                    
                    if self.EventCache:
                        PickleObject(ev, S.replace(".root", ""), FileOut)
 
                    if self.DataCache: 
                        self.AddSample(ev, self.Tree, self.SelfLoop, self.FullyConnect, -1) 
        if self.DataCache:
            self.MakeTrainingSample(self.ValidationSampleSize)
            
            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            for i in self.DataContainer:
                address = hex(id(self.DataContainer[i]))
                Exp.ExportEventGraph(self.DataContainer[i].to("cpu"), str(address), self._DataLoaderDir)
                self.Notify("!!DUMPED EVENT " + str(int(self.DataContainer[i].i)) + "/" + str(len(self.DataContainer)))
            
            RandomTestSample = {}
            for i in self.TrainingSample:
                self.TrainingSample[i] = [str(hex(id(k))) for k in self.TrainingSample[i]]
                RandomTestSample[i] = random.sample(self.TrainingSample[i], 10) 

            for i in self.ValidationSample:
                self.ValidationSample[i] = [str(hex(id(k))) for k in self.ValidationSample[i]]

            PickleObject(self.FileTraces, "FileTraces", self._DataLoaderDir + "FileTraceDump")
            PickleObject(self.TrainingSample, "TrainingSample", self._DataLoaderDir + "FileTraceDump") 
            PickleObject(self.ValidationSample, "ValidationSample", self._DataLoaderDir + "FileTraceDump")
            PickleObject(RandomTestSample, "RandomTestSample", self._DataLoaderDir + "Test")

    def __Optimizer(self):
        self.__CheckSettings(True)
            
        # Check if the model can be backed up 
        samples = UnpickleObject("RandomTestSample", self._DataLoaderDir + "Test")
        for i in samples:
            if i == 0:
                continue
            self.epoch = 0
            self.Sample = self.RecallFromCache(samples[i][0], self.CacheDir)
            self.Sample.i = torch.tensor([1], device = self.Device)
            self.Sample.to(self.Device)
            break
        if isinstance(self.Model, str) or self.Model == None:
            self.Warning("Need to specify a valid model.")
            return 
        self.InitializeModel()
        self.ExportModel(self.Sample)
       
        # Test the model for all node types 
        for i in samples:
            if i == 0:
                continue
            self.Sample = self.RecallFromCache(samples[i][0], self.CacheDir)
            self.Sample.i = torch.tensor([1], device = self.Device)
            self.Sample.to(self.Device)
            self.MakePrediction(self.Sample)
        self.Notify("PASSED RANDM n-Node TEST.")
        self.KFoldTraining()

    def Launch(self):
        self.__CheckSettings()
        self.__BuildStructure()
        self.__BuildSampleCache()
        if self.Model != None:
            self.__Optimizer()
        os.chdir(self._launchDir)
