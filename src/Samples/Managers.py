import AnalysisTopGNN
from .ROOTFile import ROOTFile
from .Event import Event
from AnalysisTopGNN.Tools import Tools

class SampleContainer(AnalysisTopGNN.Notification.SampleContainer):

    def __init__(self):
        self.EventInfo = {}
        self.ROOTInfo = {}
        self.Events = {}
        self.VerboseLevel = 3
        self.EventStart = 0
        self.EventStop = None
    
    def Initialize(self, Caller):
        self.Caller = Caller
        self.EventInfo[Caller] = {}

    def Add(self, InptDic):
        self.EventInfo[self.Caller] |= InptDic

    def AddSamples(self, Directory, Files):
        self.EventInfo[self.Caller] |= {Directory : Files}
        self.RegisteredDirectory(Directory)

    def AddROOTFile(self, Obj):
        F = ROOTFile()
        F.Trees += Obj.Trees
        F.Branches += Obj.Branches
        F.Leaves += Obj.Leaves
        F.EventIndex |= {t : -1 for t in F.Trees}
        self.ROOTInfo[Obj.ROOTFile] = F
    
    def AddEvent(self, InptEvent):
        if InptEvent.Tree not in self.EventInfo[self.Caller]:
            self.EventInfo[self.Caller][InptEvent.Tree] = 0

        it = self.EventInfo[self.Caller][InptEvent.Tree] 
        if self.EventStart > it:
            self.EventInfo[self.Caller][InptEvent.Tree] += 1
            return True
        if self.EventStop < it and self.EventStop != None:
            self.EventInfo[self.Caller][InptEvent.Tree] += 1
            return True
        if it not in self.Events:
            self.Events[it] = Event()
        self.Events[it].Trees[InptEvent.Tree] = InptEvent
        self.Events[it].iter = InptEvent.iter
        self.EventInfo[self.Caller][InptEvent.Tree] += 1
        return False

class SampleTracer(Tools):

    def __init__(self):
        pass

    def BeginTrace(self):
        self.Tracer = SampleContainer()
        self.Tracer.Initialize(self.Caller)
        self.Tracer.VerboseLevel = self.VerboseLevel
        self.Tracer.EventStart = self.EventStart
        self.Tracer.EventStop = self.EventStop

    def AddInfo(self, key, Inpt):
        self.Tracer.Add({key : Inpt})

    def AddSamples(self, Directory, Files):
        self.Tracer.AddSamples(Directory, Files)


        
