from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Settings
from AnalysisTopGNN.IO import HDF5, PickleObject, UnpickleObject

class Analysis(GraphGenerator):

    def __init__(self):
        Settings.__init__(self) 

        self.Event = None 
        self.EventGraph = None
        
        self.Caller = "ANALYSIS"

        self._SampleMap= {}

        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}
        
        SampleTracer.__init__(self, self)
        self.Settings = Settings()

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
    
    def __DumpCache(self, events, outdir):
        gener = {t.Filename : t for t in events.values() if t.Filename not in self._Cache}
        if len(gener) == 0:
            return 
        
        if self.DumpHDF5:
            hdf = HDF5()
            hdf.VerboseLevel = self.VerboseLevel
            hdf.Threads = self.Threads
            hdf.chnk = self.chnk
            hdf.Filename = "Events"
            hdf.MultiThreadedDump(events, outdir)
        
        if self.DumpPickle:
            PickleObject(events, outdir + "/Events")

    def __EventGenerator(self, InptMap, output, name):
        if self.EventCache == False:
            return {}
        
        for f in self.DictToList(InptMap):
            tmp = f.split("/")
            outdir = output + "/" + name + "/" + tmp[-1].split(".")[0]
            self.mkdir(outdir)
                
            ev = EventGenerator({"/".join(tmp[:-1]) : tmp[-1]}, self.EventStart, self.EventStop)
            ev.Event = self.Event
            ev.EventStart = self.EventStart
            ev.EventStop = self.EventStop
            ev.VerboseLevel = self.VerboseLevel
            ev._PullCode = self._PullCode
            ev.SpawnEvents()
            ev.CompileEvent()
            if self._PullCode:
                self += ev
                return {}
            self.__DumpCache({i.Filename : i for i in ev.Tracer.Events.values()}, outdir)

            self += ev
            
            events = self.Tracer.Events
            self.Tracer.Events = {}
            PickleObject(self, self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + name)
            self.Tracer.Events = events

    def __GraphGenerator(self, InptMap, output, name):
        if self.DataCache == False:
            return {}
        
        GraphAttribute = self.GraphAttribute
        NodeAttribute = self.NodeAttribute
        EdgeAttribute = self.EdgeAttribute
            
        events = {}
        for i in self:
            file = self.HashToROOT(i.Filename)
            file = file.split("/")[-1].replace(".root", "")
            if file not in events:
                events[file] = {}
            events[file][self.HashToIndex(i.Filename)] = i
        
        for file in events:
            self.mkdir(output + "/" + name + "/" + file)
            gr = GraphGenerator()
            gr.EventGraph = self.EventGraph
            gr.Tracer.Events |= events[file]
            gr.GraphAttribute |= self.GraphAttribute
            gr.NodeAttribute |= self.NodeAttribute
            gr.EdgeAttribute |= self.EdgeAttribute
            gr.CompileEventGraph()
            if self._PullCode:
                self += gr
                return {}
            self.__DumpCache(gr.Tracer.Events, output + "/" + name + "/" + file)
            
            self += gr
            self.GraphAttribute = {}
            self.NodeAttribute = {}
            self.EdgeAttribute = {} 

            evnt = self.Tracer.Events
            self.Tracer.Events = {}
            PickleObject(self, self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + name)
            self.Tracer.Events = evnt

            self.GraphAttribute = GraphAttribute
            self.NodeAttribute = NodeAttribute
            self.EdgeAttribute = EdgeAttribute

    def __BuildSampleDirectory(self, InputMap, CacheType, build):
        if build == False:
            return        
        for name in InputMap:
            output = self.OutputDirectory + "/" + self.ProjectName + "/" + CacheType 
            self.mkdir(output)
            self.__EventGenerator(InputMap[name], output, name)
            self.__GraphGenerator(InputMap[name], output, name)

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
                for j in SampleDirectory:
                    hdf.Filename = "Events"
                    hdf.MergeHDF5(root + "/" + j.split("/")[-1]) 

            tracer = self.ProjectName + "/Tracers/" + Name
            if self.IsFile(tracer + ".pkl"): 
                self += SampleTracer(UnpickleObject(tracer + ".pkl"))
            else:
                return 

    def Launch(self):
        self.__BuildRootStructure()
        if self._PullCode == False:
            self.__SearchAssets(self._SampleMap, "EventCache")
            self._Cache = {t.Filename : t for t in self.Tracer.Events.values()}
        self.__BuildSampleDirectory(self._SampleMap, "EventCache", self.EventCache)
    
        if self._PullCode == False:
            self.__SearchAssets(self._SampleMap, "DataCache")
            self._Cache = {t.Filename : t for t in self.Tracer.Events.values() if t.Compiled}
        self.__BuildSampleDirectory(self._SampleMap, "DataCache", self.DataCache)
       
        if self._PullCode == False:
            self.__SearchAssets(self._SampleMap, "EventCache")
            self.__SearchAssets(self._SampleMap, "DataCache")
        self.MakeCache() 

        self.Settings.DumpSettings(self)

