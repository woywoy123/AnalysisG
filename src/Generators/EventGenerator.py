from AnalysisTopGNN.IO import File
from AnalysisTopGNN.Notification import EventGenerator
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Samples.Event import EventContainer
from AnalysisTopGNN.Tools import Threading

class EventGenerator(EventGenerator, SampleTracer):
    def __init__(self, InputDir = None, EventStart = 0, EventStop = None):
        self.Caller = "EVENTGENERATOR"
        self.InputDirectory = InputDir
        self.EventStart = EventStart
        self.EventStop = EventStop
        self.Event = None
        self.VerboseLevel = 3
        self.chnk = 50
        self.Threads = 12
   
    def __GetEvent(self):
        if "__init__" in self.Event.__dict__:
            self.Event = self.Event()
        _, evnt = self.GetObjectFromString(self.Event.__module__, type(self.Event).__name__)
        return evnt()

    def __AddEvent(self, File, val = False):
        if val:
            
            tr = self.Tracer
            if File._Tree not in tr.EventInfo[self.Caller]:
                tr.EventInfo[self.Caller][File._Tree] = 0
            it = tr.EventInfo[self.Caller][File._Tree]
            
            if tr.EventStart > it:
                tr.EventInfo[self.Caller][File._Tree] += 1
                return False

            if tr.EventStop != None and tr.EventStop < it:
                tr.EventInfo[self.Caller][File._Tree] += 1
                return True

            if it not in tr.Events:
                tr.Events[it] = EventContainer()
                tr.Events[it].Filename = File.ROOTFile

            EventObj = self.__GetEvent()
            EventObj._Store = val
            EventObj.Tree = File._Tree
            EventObj._SampleIndex = self._iter[File._Tree]
            tr.ROOTInfo[File.ROOTFile].MakeHash(self._iter[File._Tree]) 
            tr.Events[it].Trees[File._Tree] = EventObj
            tr.Events[it].EventIndex = it
            tr.EventInfo[self.Caller][File._Tree] += 1
            return False

        for i in File:
            if self.__AddEvent(File, i):
                return True
            self._iter[File._Tree] += 1

    def SpawnEvents(self):
        self.CheckSettings()
        self.CheckEventImplementation()
        self.BeginTrace()

        Path = self.Event.__module__ + "." + self.Event.__name__
        self.AddInfo("Name", [self.Event.__name__])
        self.AddInfo("Module", [self.Event.__module__])
        self.AddInfo("Path", [Path])
        self.AddInfo("EventCode", [self.GetSourceFile(self.Event)])
        obj = self.__GetEvent()

        particles = []
        for p in obj.Objects:
            particles.append(self.GetSourceFile(obj.Objects[p]))
        particles = list(set(particles))
        self.AddInfo("ParticleCode", particles)

        self.Files = self.ListFilesInDir(self.InputDirectory, extension = ".root") 
        self.CheckROOTFiles() 
        
        for i in self.Files:
            self.AddSamples(i, self.Files[i])
        
        for F in self.DictToList(self.Files):
            F_i = File(F, self.Threads)
            F_i.Tracer = self.Tracer
            F_i.Trees += obj.Trees
            F_i.Branches += obj.Branches
            F_i.Leaves += obj.Leaves 
            F_i.ValidateKeys()
            self._iter = {}
            for tr in F_i.Trees:
                self._iter[tr] = 0
                F_i.GetTreeValues(tr)
                if self.__AddEvent(F_i):
                    return 
        
        self.CheckSpawnedEvents()

    def CompileEvent(self, ClearVal = True):
        
        def function(inp):
            out = []
            for k in inp:
                k.MakeEvent(ClearVal)
                out.append(k)
            return out
       
        TH = Threading(list(self.Tracer.Events.values()), function, threads = self.Threads, chnk_size = self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        
        for ev in TH._lists:
            self.Tracer.Events[ev.EventIndex] = ev
