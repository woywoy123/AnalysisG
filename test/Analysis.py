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
            return 
        for k in self.DictToList(InptMap):
            for f in k:
                tmp = f.split("/")
                self.mkdir(output + "/" + tmp[-1].split(".")[0])

                ev = EventGenerator({"/".join(tmp[:-1]) : tmp[-1]}, self.EventStart, self.EventStop)
                ev.Event = self.Event
                ev.EventStart = self.EventStart
                ev.EventStop = self.EventStop
                ev.VerboseLevel = self.VerboseLevel
                ev.SpawnEvents()
                ev.CompileEvent()

                for i in ev:
                    print(i.Filename)

                print("----")
                self = self + ev
                for i in self:
                    print(i.Filename)



    def __GraphGenerator(self, InptMap, output):
        if self.DataCache == False:
            return 
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
            self.__EventGenerator(InputMap, output)
            self.__GraphGenerator(InputMap, output)

            Objects = {}
            for k in self:
                f = self.HashToROOT(k.Filename)
                print(f)
                print(self.HashToEvent(k.Filename), k.Filename)
                f = f.split("/")[-1]
                if f not in Objects:
                    Objects[f] = {}
                Objects[f][k.Filename] = k
             
            for f in Objects:
                name = f.split("/")[-1].split(".")[0]
                if self.DumpHDF5:
                    hdf = HDF5()
                    hdf.VerboseLevel = self.VerboseLevel
                    hdf.Threads = self.Threads
                    hdf.chnk = self.chnk
                    hdf.Filename = CacheType
                    hdf.MultiThreadedDump(Objects[f], output + "/" + name)
                    hdf.MergeHDF5(output + "/" + name)

                if self.DumpPickle:
                    PickleObject(Objects[f], output + "/" + name + "/" + CacheType)
            
            # Clear the Events because these need to pickled in the line above
            if self.DumpPickle or self.DumpHDF5:
                self.Tracer.Events = {}
            PickleObject(self.Tracer, self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + i) 

    def __SearchAssets(self, Map, CacheType):
        for Name in Map:
            root = self.ProjectName + "/" + CacheType + "/" + Name + "/"
            SampleDirectory = [root + "/" + i for i in self.ls(root)]
            if len(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".pkl")) != 0:
                inpt = ".pkl"
            elif len(self.ListFilesInDir({i : "*" for i in SampleDirectory}, ".hdf5")) != 0: 
                inpt = ".hdf5"
            else:
                return 
            Files = self.ListFilesInDir({i : "*" for i in SampleDirectory}, inpt)
            self.ImportTracer(UnpickleObject(self.OutputDirectory + "/" + self.ProjectName + "/Tracers/" + Name))
                
            events = {}
            for i in self.DictToList(Files):
                if i.endswith(".pkl"):
                    events |= UnpickleObject(i)
                    continue
                hdf = HDF5()
                hdf.Filename = i
                for tn, obj in hdf:
                    events |= {tn : obj}
            self.Tracer.Events |= events
            
            Map[Name] = [f.split("/")[-1] for f in Files]
            setattr(self, CacheType, False)

    def Launch(self):
        self.__BuildRootStructure()
        self.__SearchAssets(self._SampleMap, "EventCache")
        self.__SearchAssets(self._SampleMap, "DataCache")
        self.__BuildSampleDirectory(self._SampleMap, "EventCache", self.EventCache)
        self.__BuildSampleDirectory(self._SampleMap, "DataCache", self.DataCache)
        self.MakeCache() 

