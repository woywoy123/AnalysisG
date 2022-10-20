from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Settings
from AnalysisTopGNN.IO import HDF5, PickleObject, UnpickleObject
from AnalysisTopGNN.Notification import Analysis

class Analysis(Analysis, GraphGenerator):

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
        self.Dump = {}

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
            hdf.MergeHDF5(outdir) 
        
        if self.DumpPickle:
            PickleObject(events, outdir + "/Events")

    def __EventGenerator(self, InptMap, output, name):
        if self.EventCache == False:
            return
        
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
            ev.Threads = self.Threads
            ev.SpawnEvents()
            ev.CompileEvent()
            self += ev 
            if self._PullCode:
                return 
         
            self.__DumpCache({i.Filename : i for i in ev.Tracer.Events.values()}, outdir)
            
            if name not in self.Dump:
                self.Dump[name] = ev
            else:
                self.Dump[name] += ev

    def __GraphGenerator(self, InptMap, output, name):
        if self.DataCache == False:
            return 
        
        GraphAttribute = self.GraphAttribute
        NodeAttribute = self.NodeAttribute
        EdgeAttribute = self.EdgeAttribute
        
        gr = GraphGenerator()
        gr.VerboseLevel = self.VerboseLevel
        gr.EventGraph = self.EventGraph
        if self._PullCode == False:
            gr += InptMap
        gr.GraphAttribute |= self.GraphAttribute
        gr.NodeAttribute |= self.NodeAttribute
        gr.EdgeAttribute |= self.EdgeAttribute
        gr.EventStop = self.EventStop
        gr.EventStart = self.EventStart
        gr._PullCode = self._PullCode
        gr.CompileEventGraph()
        self += gr        
        if self._PullCode:
            return 
        
        events = {} 
        for i in gr:
            file = gr.HashToROOT(i.Filename)
            file = file.split("/")[-1].replace(".root", "")
            if file not in events:
                events[file] = {}
            events[file][i.Filename] = i
        
        for file in events:
            self.mkdir(output + "/" + name + "/" + file)
            self.__DumpCache(events[file], output + "/" + name + "/" + file)
        
        if name not in self.Dump:
            self.Dump[name] = gr
        else:
            self.Dump[name] += gr

    def __SearchAssets(self, Map, CacheType, Name):
        root = self.ProjectName + "/" + CacheType + "/" + Name + "/"
        SampleDirectory = [root + "/" + i for i in self.ls(root)]
        Files = []
        Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".pkl"))
        Files += self.DictToList(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".hdf5"))

        self.FoundCache(root, Files)
        if len(Files) == 0 or self._PullCode:
            return False
        
        tracer = self.ProjectName + "/Tracers/" + Name
        if self.IsFile(tracer + ".pkl"): 
            Tracer = SampleTracer(UnpickleObject(tracer + ".pkl"))
        else:
            self.MissingTracer(root)
            return False

        for i in Files:
            if i.endswith(".pkl"): 
                Tracer.Tracer.Events |= {t.Filename : t for t in UnpickleObject(i).values()}
            if i.endswith(".hdf5"):
                hdf = HDF5()
                hdf.Filename = i
                Tracer.Tracer.Events |= {n : t for n, t in hdf}
        Map[Name] = Tracer
        return True
    
    def Launch(self):
        self.StartingAnalysis()
        self.__BuildRootStructure()
        self._CacheEvent = {}
        self._CacheEvent |= self._SampleMap
        
        self._CacheEventData = {}
        self._CacheEventData |= self._SampleMap

        output = self.OutputDirectory + "/" + self.ProjectName + "/"
        for i in self._SampleMap:

            self._Cache = {}
            self.mkdir(output + "EventCache")
            if self.__SearchAssets(self._CacheEvent, "EventCache", i):
                self._Cache |= {t.Filename : t for t in self._CacheEvent[i].Tracer.Events.values()}
            self.__EventGenerator(self._SampleMap[i], output + "/EventCache", i)
            
            if self.EventCache and self._PullCode == False: 
                self.Dump[i].Tracer.Events = {}
                PickleObject(SampleTracer(self.Dump[i]), output + "Tracers/" + i)

            if self.DataCache == False:
                continue
            
            self._Cache = {}
            self.mkdir(output + "DataCache")
            if self.__SearchAssets(self._CacheEventData, "DataCache", i):
                self._Cache = {t.Filename : t for t in self._CacheEventData[i].Tracer.Events.values() if t.Compiled}
            self.__GraphGenerator(self._CacheEvent[i], output + "/DataCache", i)
            
            if self._PullCode == False:
                self.Dump[i].Tracer.Events = {}
                PickleObject(SampleTracer(self.Dump[i]), output + "Tracers/" + i)

        self.Settings.DumpSettings(self)

