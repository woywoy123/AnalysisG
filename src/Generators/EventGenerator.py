from AnalysisTopGNN.IO import File
from AnalysisTopGNN.Samples.Event import EventContainer
from AnalysisTopGNN.Samples.Managers import SampleTracer

from AnalysisTopGNN.Notification import EventGenerator
from AnalysisTopGNN.Tools import Threading, Tools
from .Settings import Settings

class EventGenerator(EventGenerator, Settings, Tools, SampleTracer):
    def __init__(self, InputDir = False, EventStart = 0, EventStop = None):

        self.Caller = "EVENTGENERATOR"
        Settings.__init__(self)
        SampleTracer.__init__(self)

        if isinstance(InputDir, dict):
            self.InputDirectory |= InputDir
        else:
            self.InputDirectory = InputDir

    def SpawnEvents(self):
        def PopulateEvent(eventDict, indx, F):
            EC = EventContainer()
            for tr in eventDict: 
                EC.Trees[tr] = self.CopyInstance(self.Event)
                EC.Trees[tr]._Store = eventDict[tr]
                EC.Trees[tr].Tree = tr
                EC.Trees[tr]._SampleIndex = indx
                EC.Filename = F
            EC.EventIndex = indx
            return EC
        
        self.CheckSettings()
        self.CheckEventImplementation()

        self.AddCode(self.Event)
        obj = self.CopyInstance(self.Event)
        for p in obj.Objects:
            self.AddCode(obj.Objects[p])
        
        if self._dump:
            return self

        self.Files = self.ListFilesInDir(self.InputDirectory, extension = ".root") 
        self.CheckROOTFiles()  
        
        it = -1
        for F in self.DictToList(self.Files):
            F_i = File(F, self.Threads)
            F_i.Trees += obj.Trees
            F_i.Branches += obj.Branches
            F_i.Leaves += obj.Leaves 
            F_i.ValidateKeys()
            indx = -1
            for ev in F_i:
                indx += 1
                it += 1
                if self.EventStart < it and self.EventStart != -1:
                    continue
                self.EventStart = -1
                if self.EventStop != None and self.EventStop < it:
                    break
                event = PopulateEvent(ev, indx, F)
                self.AddROOTFile(F, event)
        self.CheckSpawnedEvents()

    def CompileEvent(self, ClearVal = True):
        
        def function(inp):
            out = []
            for k in inp:
                k.MakeEvent(ClearVal)
                out.append(k)
            return out
        
        if self._dump:
            return self

        TH = Threading(self.SampleContainer.list(), 
                       function, 
                       threads = self.Threads, 
                       chnk_size = self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        for ev in TH._lists:
            self.SampleContainer[ev.Filename] = ev
