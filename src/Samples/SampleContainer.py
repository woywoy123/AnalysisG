from AnalysisTopGNN.Notification import SampleContainer
from .Event import Event

class SampleContainer(SampleContainer):

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
        F.Filename = Obj.ROOTFile
        self.ROOTInfo[Obj.ROOTFile] = F
    
    def AddEvent(self, InptEvent, Filename):
        if InptEvent.Tree not in self.EventInfo[self.Caller]:
            self.EventInfo[self.Caller][InptEvent.Tree] = 0

        it = self.EventInfo[self.Caller][InptEvent.Tree]
        if self.EventStart > it:
            self.EventInfo[self.Caller][InptEvent.Tree] += 1
            return False

        if self.EventStop < it and self.EventStop != None:
            self.EventInfo[self.Caller][InptEvent.Tree] += 1
            return True

        if it not in self.Events:
            self.Events[it] = Event()
            self.Events[it].Filename = Filename
        self.ROOTInfo[Filename].MakeHash(it) 
        self.Events[it].Trees[InptEvent.Tree] = InptEvent
        self.Events[it].EventIndex = it
        self.EventInfo[self.Caller][InptEvent.Tree] += 1
        return False
    


