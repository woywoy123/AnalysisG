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

        if isinstance(InputDir, dict):
            self.InputDirectory |= InputDir
        else:
            self.InputDirectory = InputDir
   
    def SpawnEvents(self):
        def PopulateEvent(eventDict, indx):
            EC = EventContainer()
            for tr in EventDict 
                EC.Tree[tr] = self.CopyInstance(self.Event)
                EC.Tree[tr]._Store = EventDict[tr]
                EC.Tree[tr].Tree = tr
                EC.Tree[tr]._SampleIndex = indx
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
        
        
        for F in self.DictToList(self.Files):
            F_i = File(F, self.Threads)
            F_i.Trees += obj.Trees
            F_i.Branches += obj.Branches
            F_i.Leaves += obj.Leaves 
            F_i.ValidateKeys()
            indx = 0
            for ev in F_i:
                if self.EventStart <= indx:
                    indx += 1
                    continue
                if self.EventStop != None and self.EventStop-1 <= indx:
                    break
                event = PopulateEvent(ev, indx)

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

        TH = Threading(list(self.Tracer.Events.values()), function, threads = self.Threads, chnk_size = self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        for ev in TH._lists:
            self.Tracer.Events[ev.EventIndex] = ev
