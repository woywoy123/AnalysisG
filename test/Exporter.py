from AnalysisTopGNN.Tools import Tools
from AnalysisTopGNN.Samples import SampleContainer
from AnalysisTopGNN.Notification import SampleTracer

class SampleTracer(Tools, SampleTracer):

    def __init__(self, IMPRT = None):
        self.Caller = ""
        self.VerboseLevel = 3
        self.EventStart = 0
        self.EventStop = None
        self.BeginTrace(IMPRT)
        self.MakeCache()

    def BeginTrace(self, Tracer = None):
        if Tracer == None:
            self.Tracer = SampleContainer()
        if hasattr(self, "Tracer"):
            pass
        else:
            if hasattr(Tracer, "Tracer"):
                self.Tracer = Tracer.Tracer
            else:
                return self.BeginTrace()
        self.Tracer.Initialize(self.Caller)
        self.Tracer.VerboseLevel = self.VerboseLevel
        self.Tracer.EventStart = self.EventStart
        self.Tracer.EventStop = self.EventStop

    def AddInfo(self, key, Inpt):
        self.Tracer.Add({key : Inpt})

    def AddSamples(self, Directory, Files):
        self.Tracer.AddSamples(Directory, Files)
    
    def ImportTracer(self, Inpt):
        if "Tracer" in Inpt.__dict__:
            Inpt = Inpt.Tracer
        self.BeginTrace(Inpt)
  
    def FlushCache(self):
        self._HashCache = {}

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


    def __EventInfo(self, event1, event2):
        tr = self._NewTracer
       
        mod = set([event1["Module"], event2["Module"]])
        name = set([event1["Name"], event2["Name"]])
        pth = set([event1["Path"], event2["Path"]])
        code = set([event1["EventCode"], event2["EventCode"]])

        tr.AddInfo("Name", "/".join(name))
        tr.AddInfo("Module", "/".join(mod))
        tr.AddInfo("Path", "/".join(mod))
        tr.AddInfo("EventCode", "\n\n\n".join(code)) 

        if len(mod) > 1 or len(name) > 1:
            self.DifferentClassName(mod[0] + "." + name[0], mod[1] + "." + name[1]) 
        particle = list(set([event1["ParticleCode"][0] + event2["ParticleCode"][0]]))
        tr.AddInfo("ParticleCode", particle)
       
        com = list(tr.Tracer.EventInfo[tr.Caller])
        smpl1 = {k : event1[k] for k in event1 if k not in com}
        smpl2 = {k : event2[k] for k in event2 if k not in com}
        SumSmpl = set(list(smpl1) + list(smpl2))
        Combi = {}
        for i in SumSmpl:
            val = None 
            if i in smpl1:
                val = smpl1[i]

            if i in smpl2 and val != None:
                val += smpl2[i]
            elif i in smpl2:
                val = smpl2[i]
            Combi[i] = val

        for i in Combi:
            tr.AddInfo(i, Combi[i])
    
    def __MergeEvents(self, event1, event2):
        smpls = list(event1) +  list(event2) 
        root = list(event1.values()) +  list(event2.values()) 
        
        merge = {}
        for k, r in zip(smpls, root):
            print(r.Trees)
        
        # Continue here with sample merging.
        print(smpls, root)


        pass

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
        self._NewTracer = SampleTracer()
        key = "EVENTGENERATOR"
        _EventInfo1, _EventInfo2 = self.Tracer.EventInfo, other.Tracer.EventInfo
        if key in _EventInfo1 and key in _EventInfo2:
            self._NewTracer.Caller = key
            self._NewTracer.BeginTrace()
            self.__EventInfo(_EventInfo1[key], _EventInfo2[key])
        
        _ROOTInfo1, _ROOTInfo2 = self.Tracer.ROOTInfo, other.Tracer.ROOTInfo       
        _Events, _Events = self.Tracer.Events, other.Tracer.Events  
        
        self.__MergeEvents(_ROOTInfo2, _ROOTInfo1)
        
        return self._NewTracer


from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren

def TestSamples(Files):
    File1 = Files[0]
    File2 = Files[1] 

    #Ev = EventGenerator(File1) 
    #Ev.Event = Event
    #Ev.SpawnEvents()
    #Ev.CompileEvent()
    #PickleObject(Ev, "TMP1")

    #T = EventGenerator(File2) 
    #T.Event = Event
    #T.SpawnEvents()
    #T.CompileEvent()
    #PickleObject(T, "TMP2")

    #T = EventGenerator(Files) 
    #T.Event = Event
    #T.SpawnEvents()
    #T.CompileEvent()
    #PickleObject(T, "TMP3")

    ev1 = UnpickleObject("TMP1")
    ev2 = UnpickleObject("TMP2")
    ev3 = UnpickleObject("TMP3")

    x = SampleTracer(ev1)
    y = SampleTracer(ev2)
    z = SampleTracer(ev3) 

    p = sum([x, y, z])





    print(Files)

    return True 
