import AnalysisTopGNN
from .SampleContainer import SampleContainer
from AnalysisTopGNN.Tools import Tools

class SampleTracer(Tools):

    def __init__(self, IMPRT = None):
        self.Caller = ""
        self.VerboseLevel = 3
        self.EventStart = 0
        self.EventStop = None
        self.BeginTrace(IMPRT)
        self.MakeCache()

    def BeginTrace(self, Tracer = None):

        if hasattr(self, "Tracer"):
            pass
        elif Tracer == None:
            self.Tracer = SampleContainer()
        elif hasattr(Tracer, "Tracer"):
            self.Tracer = Tracer.Tracer
        else:
            return self.BeginTrace()
        self.Tracer.Initialize(self.Caller)
        self.Tracer.VerboseLevel = self.VerboseLevel
        self.Tracer.EventStart = self.EventStart
        self.Tracer.EventStop = self.EventStop
        self.MakeCache()

    def AddInfo(self, key, Inpt):
        self.Tracer.Add({key : Inpt})

    def AddSamples(self, Directory, Files):
        self.Tracer.AddSamples(Directory, Files)
    
    def ImportTracer(self, Inpt):
        if "Tracer" in Inpt.__dict__:
            Inpt = Inpt.Tracer
        if Inpt.__name__ == "SampleContainer":
            self.Tracer = Inpt
        
        self.BeginTrace(Inpt)
  
    def MakeCache(self):
        self._HashCache = {}
        self.AddDictToDict(self._HashCache, "IndexToHash")
        self.AddDictToDict(self._HashCache, "HashToEvent")
        self.AddDictToDict(self._HashCache, "HashToFile")
        self.AddDictToDict(self._HashCache, "IndexToEvent")

        ROOTInfo = self.Tracer.ROOTInfo
        Events = self.Tracer.Events
        
        self._HashCache["HashToFile"] = {j : i  for i in ROOTInfo for j in list(ROOTInfo[i]._HashMap.values())}
        self._HashCache["HashToEvent"] = {i.Filename : i for i in Events.values()}
        self._HashCache["IndexToHash"] = {i.EventIndex : i.Filename for i in Events.values()}
        self._HashCache["IndexToEvent"] = {i.EventIndex : i for i in Events.values()}

    def IndexToHash(self, index):
        self.MakeCache()
        return self._HashCache["IndexToHash"][index]
    
    def IndexToEvent(self, Index):
        self.MakeCache()
        return self._HashCache["IndexToEvent"][Index]

    def IndexToROOT(self, index):
        self.MakeCache()
        return self._HashCache["HashToFile"][self._HashCache["IndexToHash"][index]]
    
    def HashToROOT(self, _hash):
        return self._HashCache["HashToFile"][_hash] 

    def __iter__(self):
        self.MakeCache()
        ev = self._HashCache["IndexToEvent"]
        self._events = {k : ev[k] for k in ev if ev[k] != None}
        self._iter = 0
        return self

    def __next__(self):
        ev = self._events[self._iter]
        self._iter += 1
        if self._iter == len(self._events):
            raise StopIteration()
        return ev

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        self.Tracer += other.Tracer
        self.MakeCache()
        return self
    
    def __len__(self):
        self.MakeCache()
        return len(self._HashCache["IndexToEvent"])
