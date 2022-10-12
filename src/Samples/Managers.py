import AnalysisTopGNN
from .ROOTFile import ROOTFile
from .SampleContainer import SampleContainer
from AnalysisTopGNN.Tools import Tools


class SampleTracer(Tools):

    def __init__(self, IMPRT = None):
        if IMPRT != None:
            self.Tracer = IMPRT
            self.MakeCache()

    def BeginTrace(self, Tracer = SampleContainer()):
        self.Tracer = Tracer
        self.Tracer.Initialize(self.Caller)
        self.Tracer.VerboseLevel = self.VerboseLevel
        self.Tracer.EventStart = self.EventStart
        self.Tracer.EventStop = self.EventStop

    def AddInfo(self, key, Inpt):
        self.Tracer.Add({key : Inpt})

    def AddSamples(self, Directory, Files):
        self.Tracer.AddSamples(Directory, Files)

    def ImportTracer(self, Inpt):
        self.BeginTrace(Inpt)
   
    def MakeCache(self):
        if "_HashCache" not in self.__dict__:
            self._HashCache = {}
        else:
            return 
        self.AddDictToDict(self._HashCache, "IndexToHash")
        self.AddDictToDict(self._HashCache, "HashToEvent")
        self.AddDictToDict(self._HashCache, "HashToFile")
        self.AddDictToDict(self._HashCache, "IndexToEvent")

        ROOTInfo = self.Tracer.ROOTInfo
        EventInfo = self.Tracer.EventInfo
        Events = self.Tracer.Events
        for i in ROOTInfo:
            self._HashCache["IndexToHash"] |= ROOTInfo[i]._HashMap
            self._HashCache["HashToFile"] |= {ROOTInfo[i]._HashMap[idx] : ROOTInfo[i].Filename for idx in list(ROOTInfo[i]._HashMap)}
        self._HashCache["HashToEvent"] |= {self._HashCache["IndexToHash"][ev] : Events[ev] for ev in Events}
        self._HashCache["IndexToEvent"] |= Events
        
    def IndexToHash(self, index):
        self.MakeCache()
        return self._HashCache["IndexToHash"][index]
    
    def IndexToEvent(self, Index):
        self.MakeCache()
        return self._HashCache["IndexToEvent"][Index]

    def IndexToROOT(self, index):
        self.MakeCache()
        return self._HashCache["HashToFile"][self._HashCache["IndexToHash"][index]]


