#from AnalysisTopGNN.IO import File, Directories
#from AnalysisTopGNN.Tools import TemplateThreading, Threading, RecallObjectFromString

from AnalysisTopGNN.IO import File
from AnalysisTopGNN.Parameters import Parameters
from AnalysisTopGNN.Notification import EventGenerator
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.Tools import Threading

class EventGenerator(EventGenerator, SampleTracer): #, Directories, Parameters):
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
        return evnt

    def __AddEvent(self, File, val = False):
        if val:
            EventObj = self.__GetEvent()
            EventObj._Store = val
            EventObj.Tree = File._Tree
            EventObj._SampleIndex = self.Tracer.ROOTInfo[File.ROOTFile].EventIndex[EventObj.Tree]
            return self.Tracer.AddEvent(EventObj, File.ROOTFile) 
        for i in File:
            if self.__AddEvent(File, i):
                return True

    def SpawnEvents(self):
        self.CheckSettings()
        self.CheckEventImplementation()
        self.BeginTrace()

        Path = self.Event.__module__ + "." + self.Event.__name__
        self.AddInfo("Name", self.Event.__name__)
        self.AddInfo("Module", self.Event.__module__)
        self.AddInfo("Path", Path)
        self.AddInfo("EventCode", self.GetSourceFile(self.Event))
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
            for tr in F_i.Trees:
                F_i.GetTreeValues(tr)
                if self.__AddEvent(F_i):
                    return 
        
        self.CheckSpawnedEvents()

    def CompileEvent(self, SingleThread = False, ClearVal = True):
        
        def function(inp):
            out = []
            for k in inp:
                k.MakeEvent(ClearVal)
                out.append(k)
            return out
       
        if SingleThread:
            self.Threads = 1
       
        Events = list(self.Tracer.Events.values())
        TH = Threading(Events, function, threads = self.Threads, chnk_size = self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        
        for ev in TH._lists:
            self.Tracer.Events[ev.EventIndex] = ev
        self.Events = self.Tracer.Events
