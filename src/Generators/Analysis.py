from AnalysisTopGNN.IO import HDF5, PickleObject, UnpickleObject
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Notification import Analysis
from AnalysisTopGNN.Tools import Tools

from AnalysisTopGNN.Generators.GraphGenerator import GraphFeatures
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Settings
from AnalysisTopGNN.Generators import ModelEvaluator
from AnalysisTopGNN.Generators import Optimization

class Analysis(Analysis, Settings, SampleTracer, Tools, GraphFeatures):

    def __init__(self):

        self.Caller = "ANALYSIS"
        Settings.__init__(self) 
        SampleTracer.__init__(self)

    def InputSample(self, Name, SampleDirectory = None):
        if isinstance(SampleDirectory, str):
            if SampleDirectory.endswith(".root"):
                SampleDirectory = {"/".join(SampleDirectory.split("/")[:-1]) : SampleDirectory.split("/")[-1]}
            else:
                SampleDirectory = [SampleDirectory] 

        if self.AddDictToDict(self._SampleMap, Name) or SampleDirectory == None:
            return 
        if isinstance(SampleDirectory, dict):
            self._SampleMap[Name] |= self.ListFilesInDir(SampleDirectory, ".root")
            return 
        self._SampleMap[Name] |= self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".root")
   
    def EvaluateModel(self, Name, Directory, ModelInstance = None, BatchSize = None):
        if Name not in self._ModelDirectories:
            self._ModelDirectories[Name] = []
        Directory = self.AddTrailing(Directory, "/")
        self._ModelDirectories[Name] = [Directory + i for i in self.ls(Directory) if "Epoch-" in i]
        self._ModelSaves[Name] = {"ModelInstance": self.CopyInstance(ModelInstance)} 
        self._ModelSaves[Name] |= {"BatchSize" : BatchSize}

    def __BuildRootStructure(self): 
        if self.OutputDirectory:
            self.OutputDirectory = self.cd(self.OutputDirectory)
        else:
            self.OutputDirectory = self.pwd()
        self.OutputDirectory = self.RemoveTrailing(self.OutputDirectory, "/")
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName)
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName + "/Tracers")

    def __DumpCache(self, instance, outdir, Dump, Compiled):
        if Dump == False:
            return 
        if Compiled == False:
            export = {ev.Filename : ev for ev in instance if ev.Filename not in self}
        else:
            export = {ev.Filename : ev for ev in instance if self[ev.Filename].Compiled == Compiled}
        
        if len(export) == 0:
            return 
        
        self.mkdir(outdir)
        if self.DumpHDF5:
            hdf = HDF5()
            hdf.RestoreSettings(self.DumpSettings())
            hdf.Filename = "Events"
            hdf.MultiThreadedDump(export, outdir)
            hdf.MergeHDF5(outdir) 
        
        if self.DumpPickle:
            PickleObject(export, outdir + "/Events")

    def __EventGenerator(self, InptMap, name):
        
        for f in self.DictToList(InptMap):
            tmp = f.split("/")
           
            self.InputDirectory = {"/".join(tmp[:-1]) : tmp[-1]}
            ev = EventGenerator()
            ev.RestoreSettings(self.DumpSettings())
            ev.SpawnEvents()
            ev.CompileEvent()
            self.GetCode(ev)
            if self._dump:
                return 
            self.__DumpCache(ev, self.output + "/EventCache/" + name + "/" + tmp[-1].split(".")[0], self.EventCache, False)
            
            if self.EventCache:
                ROOT = ev.GetROOTContainer(f)
                ROOT.ClearEvents()
            self += ev
            
    def __GraphGenerator(self, InptMap, name):
        
        for f in self.DictToList(InptMap):
            tmp = f.split("/")
            gr = GraphGenerator()
            gr.SampleContainer.ROOTFiles[f] = self.GetROOTContainer(f)
            gr.RestoreSettings(self.DumpSettings())
            gr.CompileEventGraph()
            self.GetCode(gr)
            if self._dump:
                return 
            self.__DumpCache(gr, self.output + "/DataCache/" + name + "/" + tmp[-1].split(".")[0], self.DataCache, True)

            ROOT = gr.GetROOTContainer(f)
            ROOT.ClearEvents()
            self += gr
 
    def __GenerateTrainingSample(self):
        def MarkSample(smpl, status):
            for i in smpl:
                obj = self[i]
                obj.Train = status
        
        if self.TrainingSampleName == False:
            return 
        
        if self.IsFile(self.ProjectName + "/Tracers/TrainingSample/" + self.TrainingSampleName + ".pkl"):
            Training = UnpickleObject(self.ProjectName + "/Tracers/TrainingSample/" + self.TrainingSampleName)
            self._SampleMap = Training["SampleMap"]
            for i in self._SampleMap:
                self.__SearchAssets("DataCache", i)
            MarkSample(Training["train_hashes"], True)
            MarkSample(Training["test_hashes"], False)
            return self 
        self.EmptySampleList()
        inpt = {}
        for i in self:
            inpt[i.Filename] = i
        
        self.CheckPercentage()
        hashes = self.MakeTrainingSample(inpt, self.TrainingPercentage)
        
        hashes["train_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["train_hashes"]}
        hashes["test_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["test_hashes"]}
        hashes["SampleMap"] = self._SampleMap
        PickleObject(hashes, self.ProjectName + "/Tracers/TrainingSample/" + self.TrainingSampleName)
        self.__GenerateTrainingSample()

    def __Optimization(self):
        if self.Model == None:
            return 
        op = Optimization()
        op += self
        op.RestoreSettings(self.DumpSettings())
        op.Launch()
        self.GetCode(op)

    def __ModelEvaluator(self):
        if len(self._ModelDirectories) == 0:
            return 
        me = ModelEvaluator()
        me += self
        me.RestoreSettings(self.DumpSettings())
        me.Compile()


    def __SearchAssets(self, CacheType, Name):
        root = self.output + "/" + CacheType + "/" + Name + "/"
        SampleDirectory = [root + "/" + i for i in self.ls(root)]
        Files = []
        Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".pkl"))
        Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".hdf5"))
        if self.FoundCache(root, Files):
            return False
        if self.IsFile(self.output + "/Tracers/" + Name + ".pkl") == False:
            self.MissingTracer(self.output + "/Tracers/" + Name + ".pkl")
            return False

        SampleContainer = UnpickleObject(self.output + "/Tracers/" + Name + ".pkl")
        for i in Files:
            if i.endswith(".hdf5"):
                hdf = HDF5()
                hdf.RestoreSettings(self.DumpSettings())
                hdf.Filename = i
                events = {_hash : ev for _hash, ev in hdf}
            elif i.endswith(".pkl"):
                events = UnpickleObject(i)
            SampleContainer.RestoreEvents(events)
        self.SampleContainer += SampleContainer

    def Launch(self):
        self.StartingAnalysis()
        self.__BuildRootStructure()
        self.output = self.OutputDirectory + "/" + self.ProjectName
        
        for i in self._SampleMap:
            tracername = self.output + "/Tracers/" + i
            
            self.ResetSampleContainer()   
            if self.TrainingSampleName and (self.Event == None or self.EventGraph == None):
                search = "EventCache" if self.EventCache else False
                search = "DataCache" if self.DataCache else search
                if search == False: 
                    self.CantGenerateTrainingSample()           
                self.__SearchAssets(search, i)
                continue

            if self.Event == None and self.EventCache:
                self.NoEventImplementation()

            if self.DataCache and self.EventGraph == None:
                self.NoEventGraphImplementation()

            if self.EventCache and self.__SearchAssets("EventCache", i) == False:
                self.__EventGenerator(self._SampleMap[i], i)

                    
            if self.DataCache and self.__SearchAssets("DataCache", i):
                continue

            if self.__SearchAssets("EventCache", i) == False and self.DataCache:
                if self.Event == None:
                    self.NoEventImplementation()
                self.__EventGenerator(self._SampleMap[i], i)
            
            if self.DataCache: 
                self.__GraphGenerator(self._SampleMap[i], i)
            
            self.SampleContainer.ClearEvents()
            PickleObject(self.SampleContainer, tracername) 

        self.__GenerateTrainingSample()
        self.__Optimization() 
        self.__ModelEvaluator()

        self.WhiteSpace()



