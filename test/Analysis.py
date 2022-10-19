from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.IO import HDF5, PickleObject, UnpickleObject

class Analysis(GraphGenerator):

    def __init__(self):
        SampleTracer.__init__(self)
        self.EventCache = False
        self.DataCache = False
        self.Event = None 
        self.EventGraph = None
        self.Tree = None 
        self.Threads = 12
        self.chnk = 2
        
        self.DumpHDF5 = True 
        self.DumpPickle = True 
        self.OutputDirectory = False
        self.ProjectName = "UNTITLED"

        self.Caller = "ANALYSIS"

        self._SampleMap= {}
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}


    def InputSample(self, Name, SampleDirectory = None):
        if isinstance(SampleDirectory, str):
            SampleDirectory = [SampleDirectory] 

        if self.AddDictToDict(self._SampleMap, Name) or SampleDirectory == None:
            return 
        if isinstance(SampleDirectory, dict):
            self._SampleMap[Name] |= self.ListFilesInDir(SampleDirectory, ".root")
            return 
        self._SampleMap[Name] |= self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".root")

    def __BuildRootStructure(self): 
        if self.OutputDirectory:
            self.OutputDirectory = self.cd(self.OutputDirectory)
        else:
            self.OutputDirectory = self.pwd()
        self.OutputDirectory = self.RemoveTrailing(self.OutputDirectory, "/")
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName)
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName + "/Tracers")
     
    def __EventGenerator(self, InptMap, output):
        if self.EventCache == False:
            return {}
        Output = {}
        for f in self.DictToList(InptMap):
            tmp = f.split("/")
            self.mkdir(output + "/" + tmp[-1].split(".")[0])
            ev = EventGenerator({"/".join(tmp[:-1]) : tmp[-1]}, self.EventStart, self.EventStop)
            ev.Event = self.Event
            ev.EventStart = self.EventStart
            ev.EventStop = self.EventStop
            ev.VerboseLevel = self.VerboseLevel
            ev.SpawnEvents()
            ev.CompileEvent()
            Output[f] = ev
        return Output

    def __GraphGenerator(self, InptMap, output):
        if self.DataCache == False:
            return {}
        for k in InptMap:
            for f in InptMap[k]:
                self.mkdir(output + "/" + f)

        self.MakeCache()
        self.Tracer.Events = self._HashCache["IndexToEvent"]
        gr = GraphGenerator()
        gr.ImportTracer(self)
        gr.EventGraph = self.EventGraph
        gr.GraphAttribute |= self.GraphAttribute
        gr.NodeAttribute |= self.NodeAttribute
        gr.EdgeAttribute |= self.EdgeAttribute
        gr.CompileEventGraph()

    def __BuildSampleDirectory(self, InputMap, CacheType, build):
        if build == False:
            return 
        
        for i in InputMap:
            output = self.OutputDirectory + "/" + self.ProjectName + "/" + CacheType + "/" + i
            self.mkdir(output)
            out = {}
            out |= self.__EventGenerator(InputMap[i], output)
            out |= self.__GraphGenerator(InputMap[i], output)
            
            for k in out:
                cache = {t.Filename : t for t in self.Tracer.Events.values()}
                gener = {t.Filename : t for t in out[k].Tracer.Events.values() if t.Filename not in cache}
                out[k].Tracer.Events = gener
                file = k.split("/")[-1].split(".")[0]
                if len(gener) == 0:
                    continue
 
                if self.DumpHDF5:
                    hdf = HDF5()
                    hdf.VerboseLevel = self.VerboseLevel
                    hdf.Threads = self.Threads
                    hdf.chnk = self.chnk
                    hdf.Filename = "Events"
                    hdf.MultiThreadedDump(out[k].Tracer.Events, output + "/" + file)
                
                if self.DumpPickle:
                    PickleObject(out[k].Tracer.Events, output + "/" + file + "/Events")

                events = out[k].Tracer.Events
                out[k].Tracer.Events = {}
                PickleObject(out[k].Tracer, self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + i)
                out[k].Tracer.Events = events 
                self += out[k]



    def __SearchAssets(self, Map, CacheType):
        for Name in Map:
            root = self.ProjectName + "/" + CacheType + "/" + Name + "/"
            SampleDirectory = [root + "/" + i for i in self.ls(root)]
            Files = []
            Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".pkl"))
            Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".hdf5"))
            if len(Files) == 0:
                return 
            
            for i in Files:
                if i.endswith(".pkl"): 
                    self.Tracer.Events |= {t.Filename : t for t in UnpickleObject(i).values()}
                if i.endswith(".hdf5"):
                    hdf = HDF5()
                    hdf.Filename = i
                    self.Tracer.Events |= {n : t for n, t in hdf}
            
            if len([i for i in Files if i.endswith(".hdf5")]) > 1:
                hdf = HDF5()
                hdf.Filename = "Events"
                for j in SampleDirectory:
                    hdf.MergeHDF5(root + "/" + j.split("/")[-1]) 

            tracer = self.ProjectName + "/Tracers/" + Name
            if self.IsFile(tracer + ".pkl"): 
                self += SampleTracer(UnpickleObject(tracer + ".pkl"))
            else:
                return 

    def Launch(self):
        self.__BuildRootStructure()
        self.__SearchAssets(self._SampleMap, "EventCache")
        self.__SearchAssets(self._SampleMap, "DataCache")
        self.__BuildSampleDirectory(self._SampleMap, "EventCache", self.EventCache)
        self.__BuildSampleDirectory(self._SampleMap, "DataCache", self.DataCache)
        self.__SearchAssets(self._SampleMap, "EventCache")
        self.__SearchAssets(self._SampleMap, "DataCache")
        self.MakeCache() 

