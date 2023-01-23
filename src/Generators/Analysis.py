from AnalysisTopGNN.IO import HDF5, Pickle, PickleObject, UnpickleObject
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Notification import Analysis_
from AnalysisTopGNN.Tools import Tools

from AnalysisTopGNN.Generators.GraphGenerator import GraphFeatures
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Settings
from AnalysisTopGNN.Generators import ModelEvaluator
from AnalysisTopGNN.Generators import Optimization

class Analysis(Analysis_, Settings, SampleTracer, GraphFeatures, Tools):

    def __init__(self):

        self.Caller = "ANALYSIS"
        Settings.__init__(self) 
        SampleTracer.__init__(self)

    def InputSample(self, Name, SampleDirectory = None):
        if self._launch == False:
            self._InputValues.append({"INPUTSAMPLE" : {"Name" : Name, "SampleDirectory" : SampleDirectory}})
            return  

        if isinstance(SampleDirectory, str):
            if SampleDirectory.endswith(".root"):
                SampleDirectory = {"/".join(SampleDirectory.split("/")[:-1]) : SampleDirectory.split("/")[-1]}
            else:
                SampleDirectory = [SampleDirectory] 
        
        if self.AddDictToDict(self._SampleMap, Name) or SampleDirectory == None:
            if len(self._SampleMap[Name]) == 0:
                smple = UnpickleObject(self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + Name)
                self._SampleMap[Name] = {}
                for f in smple.ROOTFiles:
                    fdir = "/".join(f.split("/")[:-1])
                    self.AddListToDict(self._SampleMap[Name], fdir)
                    self._SampleMap[Name][fdir].append(f.split("/")[-1])
            return 
        if isinstance(SampleDirectory, dict):
            self._SampleMap[Name] |= self.ListFilesInDir(SampleDirectory, ".root")
            return 
        self._SampleMap[Name] |= self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".root")

    def EvaluateModel(self, Directory, ModelInstance = None, BatchSize = None):
        
        Directory = self.abs(self.AddTrailing(Directory, "/"))
        if self._launch == False:
            self._InputValues.append({"EVALUATEMODEL" : {"Directory": Directory, "ModelInstance" : ModelInstance, "BatchSize": BatchSize}})
            return  

        Name = Directory.split("/")[-1]
        if Name in self._ModelDirectories:
            self.ModelNameAlreadyPresent(Name)
            return 
        if len(self.ls(Directory)) == 0:
            self.InvalidOrEmptyModelDirectory()
            return 

        self._ModelDirectories[Name] = [self.abs(Directory + "/" + i) for i in self.ls(Directory) if "Epoch-" in i]
        self._ModelSaves[Name] = {"ModelInstance": ModelInstance} 
        self._ModelSaves[Name] |= {"BatchSize" : BatchSize}

    def __BuildRootStructure(self): 
        self.OutputDirectory = self.RemoveTrailing(self.OutputDirectory, "/")
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName)
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName + "/Tracers")

    def __DumpCache(self, instance, outdir, Compiled):
        
        if Compiled == False:
            export = {ev.Filename : ev for ev in instance if self[ev.Filename] != ""}
        else:
            export = {ev.Filename : ev for ev in instance if self[ev.Filename].Compiled == Compiled}
        
        if len(export) == 0:
            return 
        
        self.mkdir(outdir)
        if self.DumpHDF5:
            hdf = HDF5()
            hdf.RestoreSettings(self.DumpSettings())
            hdf.MultiThreadedDump(export, outdir)
        
        if self.DumpPickle:
            pkl = Pickle()
            pkl.RestoreSettings(self.DumpSettings())
            pkl.MultiThreadedDump(export, outdir)

    def __EventGenerator(self, name, filedir):
        if self.Event == None:
            return 
        ev = EventGenerator()
        ev.RestoreSettings(self.DumpSettings())
        ev.InputDirectory = {"/".join(filedir[:-1]) : filedir[-1]}
        ev.SpawnEvents()
        ev.CompileEvent()
        self.GetCode(ev)
        self += ev 
        
        if self.EventCache == False:
            return 
        self.__DumpCache(ev, self.output + "/EventCache/" + name + "/" + filedir[-1], False)
            
    def __GraphGenerator(self, name, filedir):
        if self.EventGraph == None:
            return 
        
        direc = "/".join(filedir)
        gr = GraphGenerator()
        gr.SampleContainer.ROOTFiles[direc] = self.GetROOTContainer(direc)
        gr.RestoreSettings(self.DumpSettings())
        gr.CompileEventGraph()
        self.GetCode(gr)
        self += gr 

        if self.DataCache == False:
            return 
        self.__DumpCache(gr, self.output + "/DataCache/" + name + "/" + filedir[-1], True)

    def __GenerateEvents(self, InptMap, name):
        dc = self.__SearchAssets("DataCache", name)
        ec = self.__SearchAssets("EventCache", name)
        
        for f in self.DictToList(InptMap):
            tmp = f.split("/")
            
            if self.EventCache and ec == False:
                if self.Event == None:
                    self.NoEventImplementation()
                self.__EventGenerator(name, tmp)

            if self.DataCache and dc == False:
                if self.Event == None and ec == False:
                    self.NoEventImplementation()
                elif ec == False:
                    self.__EventGenerator(name, tmp) 
                if self.EventGraph == None:
                    self.NoEventGraphImplementation()
                self.__GraphGenerator(name, tmp)

            if self.DumpHDF5 == False and self.DumpPickle == False:
                continue
            self.SampleContainer.ClearEvents()
            PickleObject(self.SampleContainer, self.output + "/Tracers/" + name) 


    def __GenerateTrainingSample(self):
        def MarkSample(smpl, status):
            for i in smpl:
                obj = self[i]
                obj.Train = status
        
        if self.TrainingSampleName == False:
            return 
        
        self.EmptySampleList()
        if self.Training == False:
            self.CheckPercentage()
            inpt = {i.Filename : i for i in self}
            hashes = self.MakeTrainingSample(inpt, self.TrainingPercentage)
            
            hashes["train_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["train_hashes"]}
            hashes["test_hashes"] = {hsh : self.HashToROOT(hsh) for hsh in hashes["test_hashes"]}
            hashes["SampleMap"] = self._SampleMap

            Name = self.TrainingSampleName if self.TrainingSampleName else "UNTITLED"
            PickleObject(hashes, self.output + "/Tracers/TrainingSample/" + Name)
            self.Training = hashes

        MarkSample(self.Training["train_hashes"], True)
        MarkSample(self.Training["test_hashes"], False)

    def __Optimization(self):
        if self.Model == None:
            return
        op = Optimization()
        op += self
        op.RestoreSettings(self.DumpSettings())
        op.Launch()
        self.GetCode(op)

    def __ModelEvaluator(self):
        if len(self._ModelDirectories) == 0 and self.PlotNodeStatistics == False:
            return 
        me = ModelEvaluator()
        me += self
        me.RestoreSettings(self.DumpSettings())
        me.Compile()

    def __SearchAssets(self, CacheType, Name):
        if isinstance(Name, list):
            for i in Name:
                self.__SearchAssets(CacheType, i)
            return 

        if CacheType == "EventCache" and self.EventCache == False:
            return 
        if CacheType == "DataCache" and self.DataCache == False:
            return 

        if self.IsFile(self.output + "/Tracers/" + Name + ".pkl") == False:
            if self.DumpPickle == False and self.DumpHDF5 == False:
                self.MissingTracer(self.output + "/Tracers/" + Name + ".pkl")
            return False
        SampleContainer = UnpickleObject(self.output + "/Tracers/" + Name + ".pkl")

        root = self.output + "/" + CacheType + "/" + Name 
        SampleDirectory = [root + "/" + i for i in self.ls(root)]
        
        _pkl = self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".pkl"))
        if len(_pkl) != 0:
            pkl = Pickle()
            pkl.Caller = self.Caller
            pkl.VerboseLevel = 0
            events = pkl.MultiThreadedReading(_pkl)
            
            self.SampleContainer += SampleContainer
            self.SampleContainer.RestoreEvents(events)
            return True

        _hdf = self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".hdf5"))
        if len(_hdf) != 0:
            hdf = HDF5()
            hdf.Caller = self.Caller
            hdf.VerboseLevel = 0
            hdf.MultiThreadedReading(_hdf)
            events = {_hash : ev for _hash, ev in hdf._name}
            
            self.SampleContainer += SampleContainer
            self.SampleContainer.RestoreEvents(events)
            return True
       
        else:
            self.NoCache(root)
            return False

    def Launch(self):
        self.__CheckSettings()
        self._launch = True 
        for i in self._InputValues:
            name = list(i)[0]
            if name == "INPUTSAMPLE":
                self.InputSample(**i["INPUTSAMPLE"])
            elif name == "EVALUATEMODEL":
                self.EvaluateModel(**i["EVALUATEMODEL"])
        
        self.StartingAnalysis()
        self.__BuildRootStructure()
        self.output = self.OutputDirectory + "/" + self.ProjectName
       
        self.Training = False
        if self.TrainingSampleName:
            fDir = self.output + "/Tracers/TrainingSample/" + self.TrainingSampleName + ".pkl" 
            self.Training = UnpickleObject(fDir) if self.IsFile(fDir) else False
            self._SampleMap = self.Training["SampleMap"] if self.Training else self._SampleMap

        if self.Model != None:
            self.DataCache = True 
        if len(self._ModelDirectories) != 0 or self.PlotNodeStatistics:
            self.DataCache = True 
        self.EventImplementationCommit() 
        
        for i in self._SampleMap:
            if self.TrainingSampleName or (self.Event == None and self.EventGraph == None) or self.FeatureTest:
                search = "EventCache" if self.EventCache else False
                search = "DataCache" if self.DataCache else search
                if search == False: 
                    self.CantGenerateTrainingSample()
                self.__SearchAssets(search, i)
                continue
            if self.DumpHDF5 or self.DumpPickle:
                self.ResetSampleContainer()   
            self.NoSamples(self._SampleMap[i], i)
            self.__GenerateEvents(self._SampleMap[i], i)
            
        self.__GenerateTrainingSample()
        self.__Optimization() 
        self.__ModelEvaluator()

        self.WhiteSpace()
    
    def __iter__(self):
        if self.SampleContainer._locked or self._launch == False:
            self.output = self.OutputDirectory + "/" + self.ProjectName
            ec = self.ls(self.output + "/EventCache")
            dc = self.ls(self.output + "/DataCache")

            trig = False
            if self.EventCache and len(ec) != 0:
                typ, dx = "EventCache", ec
                trig = True
            if self.DataCache and len(dc) != 0:
                typ, dx = "DataCache", dc
                trig = True
            if trig == False:
                typ, dx = ("DataCache", dc) if len(dc) > len(ec) else ("EventCache", ec)
            self.__SearchAssets(typ, dx)
        self._lst = [i for i in self.SampleContainer.list() if i != ""]

        if len(self._lst) == 0:
            self.NothingToIterate()
            return self

        if self.Tree == None:
            self.Tree = list(self._lst[0].Trees)[0]
        self.Lumi = 0
        return self

    def __next__(self):
        if len(self._lst) == 0:
            raise StopIteration()
        evnt = self._lst.pop(0)
        self.Lumi += float(evnt.Trees[self.Tree].Lumi)
        return evnt
