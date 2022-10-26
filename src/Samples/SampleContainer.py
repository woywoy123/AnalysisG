from AnalysisTopGNN.Notification import SampleContainer
from .ROOTFile import ROOTFile

class SampleContainer(SampleContainer):

    def __init__(self):
        self.EventInfo = {}
        self.ROOTInfo = {}
        self.Events = {}
        self.VerboseLevel = 3
        self.EventStart = 0
        self.EventStop = None
        self.__name__ = "SampleContainer"
    
    def Initialize(self, Caller):
        self.Caller = Caller
        if Caller not in self.EventInfo:
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
    
    def __add__(self, other):
        _EventInfo1, _EventInfo2 = self.EventInfo, other.EventInfo
        _Events1, _Events2 = self.Events, other.Events
        
        Event = list(_Events2.values()) + list(_Events1.values())
        trees = {}
        for i in Event:
            for k in i.Trees:
                if k not in trees:
                    trees[k] = []
                trees[k] += [i.Filename]
        
        for j in _EventInfo2:
            if j not in _EventInfo1:
                _EventInfo1[j] = _EventInfo2[j]
            for k in _EventInfo2[j]:
                if k not in _EventInfo1[j]:
                    _EventInfo1[j][k] = []
                if k in trees:
                    _EventInfo1[j][k] = len(trees[k])
                    continue
                _EventInfo1[j][k] += [t for t in _EventInfo2[j][k] if t not in _EventInfo1[j][k]]
      
        ev_self = {i.Filename : i for i in self.Events.values()}
        ev_other = {i.Filename : i for i in other.Events.values()}

        Map = {}
        Map |= ev_self
        Map |= ev_other
        Common = {i.Filename : i for i in Map.values() if i.Filename in ev_self and i.Filename in ev_other}
        for i in Common:
            if ev_self[i].Compiled:
                Map[i].Trees |= ev_self[i].Trees
                continue
            if ev_other[i].Compiled:
                Map[i].Trees |= ev_other[i].Trees
                continue
            Map[i].Trees |= ev_other[i].Trees
            Map[i].Trees |= ev_self[i].Trees
        self.Events = { i : Map[k].UpdateIndex(i) for i, k in zip(range(len(Map)), Map) }
        hashmap = {i.Filename : i.EventIndex for i in self.Events.values()}

        roots = list(self.ROOTInfo.values()) + list(other.ROOTInfo.values())
        roots = roots if isinstance(roots, list) else [roots]
        self.ROOTInfo |= {i.Filename : i for i in roots}

        for i in self.ROOTInfo:
            self.ROOTInfo[i]._HashMap |= {hashmap[k] : k for k in self.ROOTInfo[i]._HashMap.values() if k in hashmap}
        return self


