from AnalysisTopGNN.Tools import Notification, Threading
from AnalysisTopGNN.IO import WriteDirectory, Directories, PickleObject, UnpickleObject, ExportToDataScience
from AnalysisTopGNN.Generators import EventGenerator, GenerateDataLoader, Optimizer
from AnalysisTopGNN.Parameters import Parameters
import os
import random
import hashlib
import torch
import itertools
import sys


 
class Analysis(Optimizer, WriteDirectory, GenerateDataLoader, Directories, FeatureAnalysis):

    def __init__(self):
        Notification.__init__(self)
        super().__init__()
        
        self.Analysis()
        self.Caller = "Analysis"

        # ==== Hidden Internal Variables ==== #
        self._SampleMap = {}
        self._iter = 0
        self._init = False
        self._pwd = os.getcwd()
        self.pwd = self._pwd

    def InputSample(self, Name, Directory = None):
        
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

    def __CheckFiles(self, InputMap, Extension = ""):
        OutputMap = {}
        for i in InputMap:
            OutputMap[i] = [t for k in InputMap[i] for t in self.ListFilesInDir(k) if Extension in t or Extension == ""]
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
            Compiler = EventGenerator(None, self.EventStart, self.EventEnd)
            Compiler.Event = self.Event
            Compiler.VerboseLevel = self.VerboseLevel
            Compiler.Threads = self.Threads
            Compiler.Files[BaseName] = [FileName]
            Compiler.SpawnEvents()
            Compiler.CompileEvent()
            return Compiler
         
        def DataCache(EventInstance):
            if self.EventGraph == None:
                self.Fail("NO EVENTGRAPH IMPLEMENTATION GIVEN. EXITING...")
            self.TestObjectAttributes(EventInstance.Events)
            self.AddSample(EventInstance, self.Tree, self.SelfLoop, self.FullyConnect)

        def DumpHDF5(BaseDir, Filename):
            def function(inpt):
                exp = ExportToDataScience()
                exp.VerboseLevel = 0
                out = []
                for p in inpt:
                    f = str(hashlib.md5(str(BaseDir + "/" + str(int(p.i))).encode("utf-8")).hexdigest())
                    exp.ExportEventGraph(p, f, BaseDir)
                    out.append(f)
                    del p
                del exp
                return out
                    
            
            BaseDir = BaseDir.replace("./Event", "./Data")
            Start = self.FileTraces["Start"][-1]
            End = self.FileTraces["End"][-1]
            Chnk = [self.DataContainer[p] for p in range(Start, End+1)]
            TH = Threading(Chnk, function, self.Threads, self.chnk)
            TH.VerboseLevel = 0
            TH.Start()
            for p in range(len(TH._lists)):
                self.DataContainer[Start + p] = TH._lists[p]
                TH._lists[p] = None
            del TH

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
                self.ProcessSamples()
                del ev

            if self.DumpHDF5 and self.DataCache and FileName.endswith(".pkl"):
                DumpHDF5(BaseDir, FileName)
            
            # --- Do some Clean-Up if not cached.
            if self.DataCache:
                fname = list(set([i.split("/")[2] for i in InputMap]))[0]
                try:
                    Package = UnpickleObject(fname, "./DataCache/" + fname)
                except FileNotFoundError:
                    Package = {}
                Package |= self.DataContainer
                Package |= self.FileTraces
                Package |= {"DataMap" : list(InputMap)}
                PickleObject(Package, fname, "./DataCache/" + fname)

                Start, End = self.FileTraces["Start"][-1], self.FileTraces["End"][-1]
                self.DataContainer = {}
                Package = {}

    def __UpdateSampleIndex(self):
        def function(inpt):
            exp = ExportToDataScience()
            exp.VerboseLevel = 0
            out = []
            for i in inpt:
                FName = i[0].split("/")[-1]
                directory = i[1]
                it = i[2]
                out.append([it, None]) 
                
                try:
                    os.symlink(os.path.abspath(directory + "/" + FName + ".hdf5"), os.path.abspath(self.DataCacheDir + "/" + FName + ".hdf5"))
                except FileExistsError:
                    continue
                
                data = list(exp.ImportEventGraph(FName, directory).values())[0]
                data.i = torch.tensor(it)
                os.remove(directory + "/" + FName + ".hdf5")
                exp.ExportEventGraph(data, FName, directory)
                del data
            return out
        
        it = 0
        for i in self.__CheckFiles({"": ["./DataCache/"]}, ".pkl")[""]:
            Smpl = UnpickleObject(i)
            Data = { i : Smpl[i] for i in ["Tree", "Start", "End", "SelfLoop", "Samples", "Level", "DataMap"] }
            Smpl = { i : Smpl[i] for i in Smpl if i not in Data }
            F = 0
            ct = 0
            Recall = []
            for s in Smpl:
                if ct > Data["End"][F]: 
                    F+=1

                if ct == Data["Start"][F]: 
                    self.FileTraces["Start"].append(it)
                    self.FileTraces["End"].append(it)
                    
                    self.FileTraces["Samples"].append(Data["Samples"][F])
                    self.FileTraces["Tree"].append(Data["Tree"][F])
                    self.FileTraces["Level"].append(Data["Level"][F])
                    self.FileTraces["SelfLoop"].append(Data["SelfLoop"][F])

                if ct >= Data["Start"][F] and ct <= Data["End"][F]:
                    self._SampleMap[it] = Data["DataMap"][F] + "/" + Smpl[ct]
                    self.DataContainer[it] = [Smpl[ct], Data["DataMap"][F], it]
                self.FileTraces["End"][-1] = it
                ct+=1
                it+=1

            inpt = [self.DataContainer[x] for x in self.DataContainer if self.DataContainer[x] != None]
            TH = Threading(inpt, function, self.Threads, self.chnk)
            TH.Start()
            for t in TH._lists:
                self.DataContainer[t[0]] = t[1]

    def __ImportDataLoader(self):
        def function(inpt):
            exp = ExportToDataScience()
            exp.VerboseLevel = 0
            out = []
            for i in inpt:
                out.append([i[0], list(exp.ImportEventGraph(i[1], "./HDF5").values())[0]])
            return out
        inpt = [[i, self.DataContainer[i]] for i in self.DataContainer]
        TH = Threading(inpt, function, self.Threads, self.chnk)
        TH.Start()
        for i in TH._lists:
            self.DataContainer[i[0]] = i[1]

    def Launch(self):
        self.__BuildRootStructure()
        
        if self.EventCache:
            self.Caller = "Analysis(EventGenerator)"
            self.Notify("------ Checking for ROOT Files -------")
            self._SampleMap = {k : self._SampleMap[k] for k in self._SampleMap if self._SampleMap[k] != None} 
            if len(self._SampleMap) == 0:
                self.Fail("NO ROOT FILES WERE FOUND... EXITING.")
            EventMap = self.__CheckFiles(self._SampleMap)
            EventMap = self.__BuildSampleDirectory(EventMap, "EventCache")
            self.__BuildCache(EventMap) 
        
        if self.DataCache:
            self.Caller = "Analysis(DataGenerator)"
            self.Notify("------ Checking for EventCached Files -------")          
            Cache = self.__CheckFiles({"" : ["./EventCache"]})
            SMPL = list(self._SampleMap.keys())
            Cache = {"" : [ i for i in Cache[""] for j in SMPL if i.split("/")[2] == j]}
            if len(Cache[""]) == 0:
                self.Warning("NO EVENT CACHE FOUND! GENERATING EVENTS WITH CACHING...")
                self.EventCache = True
                return self.Launch() 
            
            if self.DumpHDF5:
                DirMap = {i.split("/")[2] : [] for i in Cache[""]}
                for i in Cache[""]:
                    DirMap[i.split("/")[2]].append(i)
                self.__BuildSampleDirectory(DirMap, "DataCache")
            
            DataMap = { "/".join(i.split("/")[:-1]).replace("./EventCache", "./DataCache") : i for i in Cache[""]}
            self.__BuildCache(DataMap)

        
        if self.MergeSamples: 

            self.Caller = "Analysis(Sample Merger)"
            self.MakeDir(self.DataCacheDir)
            self.__UpdateSampleIndex()
            self.MakeDir("./FileTraces")
            self._SampleMap = {i : self._SampleMap[i].split("/")[-1] for i in self._SampleMap if isinstance(i, int)}
            self.FileTraces |= {"SampleMap" : self._SampleMap}
            PickleObject(self.FileTraces, "FileTraces", "./FileTraces")
        
        if self.GenerateTrainingSample:
            self.Caller = "Analysis(Training Sample Generator)"
            try:
                self.FileTraces = UnpickleObject("FileTraces", "./FileTraces")
            except FileNotFoundError:
                self.DataCache = False
                self.EventCache = False
                self.MergeSamples = True
                return self.Launch()
            self.DataContainer = self.FileTraces["SampleMap"]
            del self.FileTraces["SampleMap"]

            self.__ImportDataLoader()
            self.MakeTrainingSample()
            self._SampleMap = {}
            for i in self.TrainingSample:
                self.TrainingSample[i] = [int(k.i) for k in self.TrainingSample[i]]
                if len(self.TrainingSample[i]) < 10:
                    t = len(self.TrainingSample[i])
                else:
                    t = 10
                self._SampleMap[i] = [k for k in random.sample(self.TrainingSample[i], t)]
            for i in self.ValidationSample:
                self.ValidationSample[i] = [int(k.i) for k in self.ValidationSample[i]]
            pgk = {}
            pgk["Validation"] = self.ValidationSample
            pgk["Test"] = self._SampleMap
            pgk["Training"] = self.TrainingSample
            PickleObject(pgk, "TrainingSample", "./FileTraces")
        
        if self.Model != None:
            self.Caller = "Analysis(Optimizer)"
            self.RunDir = "Models"
            self.CacheDir = "./HDF5"
            try:
                F = UnpickleObject("FileTraces", "./FileTraces")
            except FileNotFoundError:
                self.Fail("NO TRAINING SAMPLE HAS BEEN CREATED. FIRST MERGE THE SAMPLES AND THEN GENERATE THE TRAINING SAMPLE!")
            self.DataContainer = F["SampleMap"]
            self.FileTraces = {i : F[i] for i in F if i != "SampleMap"}
            F = UnpickleObject("TrainingSample", "./FileTraces")
            self.TrainingSample = F["Training"]
            self.ValidationSample = F["Validation"]
            self._SampleMap = F["Test"]
            exp = ExportToDataScience()
            exp.VerboseLevel = 0
            for i in self.TrainingSample:
                self.TrainingSample[i] = [self.DataContainer[k] for k in self.TrainingSample[i]]
            for i in self.ValidationSample:
                self.ValidationSample[i] = [self.DataContainer[k] for k in self.ValidationSample[i]]
            for i in self._SampleMap:
                self._SampleMap[i] = [list(exp.ImportEventGraph(self.DataContainer[k], self.CacheDir).values())[0].to(self.Device) for k in self._SampleMap[i]]
            
            self.Notify("========= Testing Sample for Model Compatibility =========")
            self.Sample = list(self._SampleMap.values())[0][0]
            self.InitializeModel()
            self.epoch = -1
            self.ExportModel(self.Sample)
            self.DefineOptimizer()
            self.GetTruthFlags(self.EdgeAttribute, "E")
            self.GetTruthFlags(self.NodeAttribute, "N")
            self.GetTruthFlags(self.GraphAttribute, "G")
            self.Debug = True
            for i in self._SampleMap:
                for k in self._SampleMap[i]:
                    self.Train(k)
            self.Debug = False

            self.Notify("========= Starting the Full Training =========")
            self.KFoldTraining()
        os.chdir(self._pwd)
