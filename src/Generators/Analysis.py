from AnalysisTopGNN.IO import HDF5, PickleObject, UnpickleObject
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Settings
from AnalysisTopGNN.Generators.GraphGenerator import GraphFeatures
from AnalysisTopGNN.Notification import Analysis
from AnalysisTopGNN.Tools import Tools

class Analysis(Analysis, Settings, SampleTracer, Tools, GraphFeatures):

    def __init__(self):

        self.Caller = "ANALYSIS"
        Settings.__init__(self) 
        SampleTracer.__init__(self)

    def __BuildRootStructure(self): 
        if self.OutputDirectory:
            self.OutputDirectory = self.cd(self.OutputDirectory)
        else:
            self.OutputDirectory = self.pwd()
        self.OutputDirectory = self.RemoveTrailing(self.OutputDirectory, "/")
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName)
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName + "/Tracers")


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
    
    def __DumpCache(self, instance, outdir, Dump, Compiled):
        if Dump == False:
            return 
        if Compiled == False:
            export = {ev.Filename : ev for ev in instance if ev.Filename not in self}
        else:
            export = {ev.Filename : ev for ev in instance if self[ev.Filename].Compiled == False}

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
            self += ev
            self.SampleContainer.ClearEvents() 

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
            self += gr
            self.SampleContainer.ClearEvents() 

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

        self.SampleContainer += UnpickleObject(self.output + "/Tracers/" + Name + ".pkl")
        for i in Files:
            if i.endswith(".hdf5"):
                hdf = HDF5()
                hdf.RestoreSettings(self.DumpSettings())
                hdf.Filename = i
                events = {_hash : ev for _hash, ev in hdf}
            elif i.endswith(".pkl"):
                events = UnpickleObject(i)
            self.SampleContainer.RestoreEvents(events)

    def Launch(self):
        self.StartingAnalysis()
        self.__BuildRootStructure()
        self.output = self.OutputDirectory + "/" + self.ProjectName
        
        for i in self._SampleMap:
            tracername = self.output + "/Tracers/" + i
            
            self.ResetSampleContainer()            
            if self.Event == None and self.EventCache:
                self.NoEventImplementation()
            elif self.EventCache:
                self.__SearchAssets("EventCache", i)
                self.__EventGenerator(self._SampleMap[i], i)
                PickleObject(self.SampleContainer, tracername) 
        
            if self.EventGraph == None and self.DataCache:
                self.NoEventImplementation()
            elif self.DataCache:
                self.__SearchAssets("DataCache", i)
                
                if len(self) == 0:
                    self.EventCache = True
                    self.DataCache = False
                    self.ResetSampleContainer()            
                    self.Launch()
                    self.DataCache = True
                    self.EventCache = False
                    self.__SearchAssets("EventCache", i)
                    self.__GraphGenerator(self._SampleMap[i], i)
                    PickleObject(self.SampleContainer, tracername) 







    # ====== Additional Interfaces. Will become separate classes later on... ===== #
    #def GenerateTrainingSample(self, TrainingPercentage):
    #    def MarkSample(smpl, status):
    #        for i in smpl:
    #            try:
    #                obj = self.HashToEvent(i)
    #            except:
    #                continue
    #            obj.Train = status

    #    if self.IsFile(self.ProjectName + "/Tracers/TrainingTestSamples.pkl"):
    #        Training = UnpickleObject(self.ProjectName + "/Tracers/TrainingTestSamples")
    #        MarkSample(Training["train_hashes"], True)
    #        MarkSample(Training["test_hashes"], False)
    #        return self 

    #    inpt = {}
    #    for i in self:
    #        inpt[i.Filename] = i
    #    hashes = self.MakeTrainingSample(inpt, TrainingPercentage)
    #    
    #    hashes["train_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["train_hashes"]}
    #    hashes["test_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["test_hashes"]}
    #    PickleObject(hashes, self.ProjectName + "/Tracers/TrainingTestSamples")
    #    self.GenerateTrainingSample(TrainingPercentage)


