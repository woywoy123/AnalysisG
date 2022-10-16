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
        if SampleDirectory == None:
            SampleDirectory = [self.ProjectName + "/EventCache/" + Name + "/" + i for i in self.ls(self.ProjectName + "/EventCache/" + Name)]
            inpt = ".pkl"
        else:
            inpt = ".root"

        if isinstance(SampleDirectory, str):
            SampleDirectory = [SampleDirectory] 

        if self.AddDictToDict(self._SampleMap, Name):
            return 
        self._SampleMap[Name] |= self.ListFilesInDir({i : "*" for i in SampleDirectory}, inpt)

    def __BuildRootStructure(self): 
        if self.OutputDirectory:
            self.OutputDirectory = self.cd(self.OutputDirectory)
        else:
            self.OutputDirectory = self.pwd()
        self.OutputDirectory = self.RemoveTrailing(self.OutputDirectory, "/")
        self.mkdir(self.OutputDirectory + "/" + self.ProjectName)
     
    def __EventGenerator(self, files):
        if self.EventCache == False:
            return 
        ev = EventGenerator(files, self.EventStart, self.EventStop)
        ev.Event = self.Event
        ev.EventStart = self.EventStart
        ev.EventStop = self.EventStop
        ev.VerboseLevel = self.VerboseLevel
        ev.SpawnEvents()
        ev.CompileEvent()
        self = self.__add__(ev)

    def __GraphGenerator(self, files):
        if self.DataCache == False:
            return 
        self.Tracer.Events |= UnpickleObject(files)
        gr = GraphGenerator()
        self = self + gr
        gr.ImportTracer(self)
        gr.EventGraph = self.EventGraph
        gr.GraphAttribute |= self.GraphAttribute
        gr.NodeAttribute |= self.NodeAttribute
        gr.EdgeAttribute |= self.EdgeAttribute
        gr.CompileEventGraph()
        print(gr.Tracer.Events)

    def __BuildSampleDirectory(self, InputMap, CacheType, build):
        if build == False:
            return 
        output = self.OutputDirectory + "/" + self.ProjectName + "/" + CacheType
        for i in InputMap:
            self.mkdir(output + "/" + i)
          
            for k in self.DictToList(InputMap[i]):
                tmp = k.split("/")
                self.mkdir(output + "/" + i + "/" + tmp[-1])
                self.__EventGenerator({"/".join(tmp[:-1]) : tmp[-1]})
                self.__GraphGenerator(k)

            Objects = {}
            for k in self:
                f = self.HashToROOT(k.Filename)
                f = f.split("/")[-1]
                if f not in Objects:
                    Objects[f] = {}
                Objects[f][k.Filename] = k
             
            for f in Objects:

                if self.DumpHDF5:
                    hdf = HDF5()
                    hdf.VerboseLevel = self.VerboseLevel
                    hdf.Filename = CacheType
                    hdf.MultiThreadedDump(Objects[f], output + "/" + i + "/" + f)
                    hdf.MergeHDF5(output + "/" + i + "/" + f)

                if self.DumpPickle:
                    PickleObject(Objects[f], output + "/" + i + "/" + f + "/" + CacheType) 


    def __UpdateSampleIndex(self):
        pass

    def __ImportDataLoader(self):
        pass

    def Launch(self):
        self.__BuildRootStructure()
        self.__BuildSampleDirectory(self._SampleMap, "EventCache", self.EventCache)
        self.__BuildSampleDirectory(self._SampleMap, "DataCache", self.DataCache)
