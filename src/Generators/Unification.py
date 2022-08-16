from AnalysisTopGNN.Tools import Notification 
from AnalysisTopGNN.IO import WriteDirectory, Directories, PickleObject, UnpickleObject, ExportToDataScience
from AnalysisTopGNN.Generators import EventGenerator, GenerateDataLoader, Optimizer
from AnalysisTopGNN.Events import Event
import os 
import sys
import random
import shutil  
import torch
import hashlib
from pathlib import Path

class Analysis(Optimizer, WriteDirectory, Directories, GenerateDataLoader, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        GenerateDataLoader.__init__(self)
        Notification.__init__(self)
        
        # ==== Initial variables ==== #
        self.Caller = "Analysis"
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
        self.DataCacheOnlyCompile = []
        self.Tree = "nominal"
        self.EventGraph = None
        self.TrainingSampleSize = 50
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}
        self.Device = "cpu" 
        self.SelfLoop = True
        self.FullyConnect = True
        self.GenerateTrainingSample = False
        self.RebuildTrainingSample = False

        # ===== Optimizer variables ====== # 
        self.LearningRate = 0.0001
        self.WeightDecay = 0.001
        self.kFold = 4
        self.Epochs = 0
        self.BatchSize = 10
        self.Model = None
        self._init = False
        self.RunName = "TESTMODEL"
        self.DefaultOptimizer = "ADAM"
        self.ONNX_Export = False
        self.TorchScript_Export = False
        self.Training = True
        self.TrainWithoutCache = False
        self.Debug = False
        self.DefaultScheduler = "ExponentialR"
        self.SchedulerParams = {"gamma" : 0.9}
        self.Scheduler = None
        
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
            try:
                self.TrainingSample = UnpickleObject("TrainingSample", self._DataLoaderDir + "FileTraceDump")
                self.ValidationSample = UnpickleObject("ValidationSample", self._DataLoaderDir + "FileTraceDump")
            except:
                self.GenerateTrainingSample = True
                self.DataCache = False
                self._SampleDir = {}
                self.Warning("NO TRAINING / VALIDATION SAMPLES FOUND! GENERATING...")
                self.__BuildSampleCache()

            tmp = self.VerboseLevel 
            self.VerboseLevel = 0 
            events = self.__CheckFiles(self._DataLoaderDir + "/HDF5")["DataLoader/HDF5"]
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
            self.CacheDir = self._DataLoaderDir + "/HDF5"
            
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
       
        tmp = {}
        for k, l in self._SampleDir.items():
            tmp[k] = self.__CheckFiles(l)
            if len(tmp[k]) == 0:
                tmp.pop(k, None)
        self._SampleDir = tmp


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
        self.Notify("CHECKING IF ANYTHING HAS BEEN CACHED.")
        cached = self.__CheckFiles(self._MainDirectory + "/" + self._EventGeneratorDir)
        if len(cached) == 0 and self.DataCache == True and len(self._SampleDir) == 0:
            self.Fail("NOTHING CACHED OR GIVEN SAMPLES NOT FOUND. EXITING...") 
        
        if self.DataCache == True and self.EventGraph == None:
            self.Fail("CAN'T CREATE SAMPLE EVENT GRAPH FOR DATALOADER. NO EVENTGRAPH FOUND. See 'src.EventTemplates.EventGraphs'")
        
        if self.DataCache == True: 
            attrs = 0
            self.NEvents = self.NEvent_Stop
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
            self._CommonDir = os.path.commonpath([i for name in self._SampleDir for i in list(self._SampleDir[name])])
            for name in self._SampleDir:
                for FileDir in self._SampleDir[name]:
                    self.MakeDir(self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, ""))
        
        if self.DataCache == True:
            self.MakeDir(self._DataLoaderDir)
            self.MakeDir(self._DataLoaderDir + "FileTraceDump")
        
        if self.GenerateTrainingSample == True:
            shutil.rmtree(self._DataLoaderDir + "Test", ignore_errors = True)
            self.MakeDir(self._DataLoaderDir + "Test")
    
    def __CheckFiles(self, Directory):
        if isinstance(Directory, str):
            Directory = [Directory]
        out = {}
        for i in Directory:
            val = self.ListFilesInDir(i)
            for p in val:
                f, key  = p.split("/")[-1], "/".join(p.split("/")[:-1]).replace("//", "/")
                if key.split("/")[-1] not in self.DataCacheOnlyCompile and len(self.DataCacheOnlyCompile) != 0:
                    self.Warning("EXCLUDING " + key + "/" + f + " FROM DATALOADER.")
                    continue
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
                return ["FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + " ".join(fail)]

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
                f = []
                if self.EventCache == False:
                    f = [k for k in Samples if k.endswith(".root")]
                else:
                    f = [k for k in Samples if k.endswith(".pkl")]
                if len(f) > 0:
                    continue

                if FileDir.split("/")[-1] not in self.DataCacheOnlyCompile and len(self.DataCacheOnlyCompile) > 0:
                    continue
                
                self.Notify("STARTING THE COMPILATION OF NEW DIRECTORY: " + FileDir)
                FileOut = self._EventGeneratorDir + "/" + name + "/" + FileDir.replace(self._CommonDir, "")
                CheckSample = True
                for S in Samples:
                    if self.EventImplementation != None and self.EventCache == True:
                        ev = EventGenerator(None, self.Verbose, self.NEvent_Start, self.NEvent_Stop)
                        ev.EventImplementation = self.EventImplementation
                        ev.Files = {FileDir : [S]}
                        ev.VerboseLevel = 0
                        ev.Threads = self.CPUThreads 
                        ev.SpawnEvents()
                        ev.CompileEvent(self.CompileSingleThread)
                    else:
                        ev = UnpickleObject(S, FileDir)
                  
                    if CheckSample and self.DataCache:
                        __TestObjectAttributes(ev.Events)
                        CheckSample = False
                    
                    if self.EventCache == True:
                        PickleObject(ev, S.replace(".root", ""), FileOut)
 
                    if self.DataCache == True: 
                        self.AddSample(ev, self.Tree, self.SelfLoop, self.FullyConnect, -1) 

        if self.DataCache:
            Start_E = self.FileTraces["Start"]
            End_E = self.FileTraces["End"]
            Sample_N = self._SampleDir[""]
            Sample_N = [l.replace("EventGenerator", "DataLoader") + "/"*(1-int(l.endswith("/"))) + j.replace(".pkl", "") for l in Sample_N for j in Sample_N[l]]
            Sample_N = [ "/".join(p.split("/")[p.split("/").index("DataLoader")+1:]) for p in Sample_N]
            self.MakeDir(self._DataLoaderDir + "/HDF5")
            
            it = 0
            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            for i in self.DataContainer:
                if self.DataContainer[i].i == Start_E[it]:
                    self._outdir = self._DataLoaderDir + "/" + Sample_N[it]
                if self.DataContainer[i].i == End_E[it]:
                    it+=1
                
                name = self._outdir + "/" + str(i)
                address = str(hashlib.md5(name.encode("utf-8")).hexdigest())
                srcdir = self._MainDirectory + "/" + self._outdir + "/" + address + ".hdf5"
                dstdir = self._MainDirectory + "/" + self._DataLoaderDir + "/HDF5/" + address + ".hdf5"
                    
                if os.path.exists(srcdir) and os.path.exists(dstdir) and self.DataCache == False:
                    self.Notify("!!EVENT NOT DUMPED, FILES EXIST " +  dstdir)
                else:
                    Exp.ExportEventGraph(self.DataContainer[i].to("cpu"), address, self._outdir)
                    self.Notify("!!DUMPED EVENT " + str(int(self.DataContainer[i].i+1)) + "/" + str(len(self.DataContainer)))
                self.DataContainer[i] = address

                try:
                    os.symlink(srcdir.replace("//", "/"), dstdir.replace("//", "/"))
                except FileExistsError:
                    pass
                except:
                    self.Fail("SYMLINKS NOT SUPPORTED!")

            prfx = ""
            if len(self.DataCacheOnlyCompile) > 0:
                prfx += "_".join(self.DataCacheOnlyCompile)
                prfx += "_"

            PickleObject(self.FileTraces, prfx + "FileTraces", self._DataLoaderDir + "FileTraceDump")
            PickleObject(self.DataContainer, prfx + "DataContainer", self._DataLoaderDir + "FileTraceDump")

        if self.GenerateTrainingSample:
            x = self.ListFilesInDir(self._DataLoaderDir + "FileTraceDump")
            Trace = [UnpickleObject(k) for k in x if "_FileTraces.pkl" in k]
            Container = [UnpickleObject(k) for k in x if "_DataContainer.pkl" in k]
            
            if len(Trace) != 0 and len(Container) != 0 and self.RebuildTrainingSample:
                self.Notify("FOUND FRAGMENTATION PIECES... NEED TO REBUILD THE DATA PROPERLY...")
                for i in Trace:
                    for key in self.FileTraces:
                        self.FileTraces[key] += i[key]
                
                start, end = [], []
                it = 0
                for k, j in zip(self.FileTraces["Start"], self.FileTraces["End"]):
                    start.append(it)
                    it += (j-k)
                    end.append(it)
                    it += 1
                self.FileTraces["End"] = end
                self.FileTraces["Start"] = start

                c = 0
                exp = ExportToDataScience()
                exp.VerboseLevel = 0
                tmp = self.Device
                self.Device = "cpu"
                p = 1
                for i in Container:
                    Data = []
                    counter = []
                    for j in i:
                        self.DataContainer[c] = i[j]
                        Data.append(i[j])
                        counter.append(torch.tensor(c))
                        c += 1
                    sets = self.RecallFromCache(Data, self._DataLoaderDir + "/HDF5/")
                    it = 0
                    for j, k in zip(sets, counter):
                        setattr(j, "i", k)
                        di = "/".join(str(Path(self._DataLoaderDir + "/HDF5/" + Data[it] + ".hdf5").resolve()).split("/")[:-1])
                        exp.ExportEventGraph(j, Data[it], di)
                        it+=1 
                        if it % 1000 == 0:
                            self.Notify("REBUILDING CONTAINER... " + str(p) + "/"  + str(len(Container)) + " - " + str(it) + "/" + str(len(sets)))
                    p+=1

                PickleObject(self.FileTraces, "FileTraces", self._DataLoaderDir + "FileTraceDump")
                PickleObject(self.DataContainer, "DataContainer", self._DataLoaderDir + "FileTraceDump")
                self.Device = tmp

            self.DataContainer = UnpickleObject("DataContainer", self._DataLoaderDir + "FileTraceDump")
            Exp = ExportToDataScience()
            
            tmp = {}
            for i in self.DataContainer:
                tmp[i] = self.DataContainer[i] 
                out = self.RecallFromCache(self.DataContainer[i], self._DataLoaderDir + "/HDF5/")
                self.DataContainer[i] = out

            RandomTestSample = {}
            self.MakeTrainingSample(self.TrainingSampleSize)
            for i in self.TrainingSample:
                self.TrainingSample[i] = [tmp[int(k.i)] for k in self.TrainingSample[i]]

                if len(self.TrainingSample[i]) < 10:
                    RandomTestSample[i] = self.TrainingSample[i]
                    continue
                RandomTestSample[i] = random.sample(self.TrainingSample[i], 10) 

            for i in self.ValidationSample:
                self.ValidationSample[i] = [tmp[int(k.i)] for k in self.ValidationSample[i]]
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
        self.Notify("PASSED RANDOM n-Node TEST.")


        self.KFoldTraining()

    def Launch(self):
        self.__CheckSettings()
        self.__BuildStructure()
        self.__BuildSampleCache()
        if self.Model != None:
            self.__Optimizer()
        os.chdir(self._launchDir)
