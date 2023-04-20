#distutils: language = c++
from ROOT cimport CySampleTracer, CyROOT, CyEvent
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef class Event:
    cdef CyEvent* ptr
    cdef public list _instance

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self):
        self._instance = []
 
    def _wrap(self, val):
        self._instance.append(val)
   
    def __getattr__(self, attr):
        return getattr(self._instance[0], attr)

    def __getstate__(self):
        out = {}
        out["_instance"] = self._instance
        return out
    
    def __setstate__(self, inpt):
        setattr(self, "_instance", inpt["_instance"])

    def __eq__(self, other) -> bool:
        if isinstance(self, str): return False
        if isinstance(other, str): return False
        return self.hash == other.hash and self.Event == other.Event

    @property
    def index(self) -> int: return self.ptr.EventIndex 

    @property
    def Graph(self) -> bool: return self.ptr.Graph
    
    @property
    def Event(self) -> bool: return self.ptr.Event

    @property
    def hash(self) -> string: return self.ptr.Hash.decode("UTF-8")

    @property 
    def ROOT(self) -> string: return self.ptr.ROOT.decode("UTF-8")

cdef class SampleTracer:
    cdef CySampleTracer* ptr
    cdef dict HashMap
    cdef vector[string] _itv
    cdef int _its
    cdef int _ite

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self.HashMap = {}

    def __dealloc__(self):
        del self.ptr

    def __init__(self):
        pass

    def __contains__(self, str key) -> bool:
        if self.ptr.ContainsROOT(key.encode("UTF-8")): return True 
        if self.ptr.ContainsHash(key.encode("UTF-8")): return True 
        return False
    
    def __getitem__(self, str key):
        cdef string i = key.encode("UTF-8"); 
        cdef vector[string] r
        cdef list out = []
        if self.ptr.ContainsROOT(i): 
            r = self.ptr.ROOTtoHashList(i)
            for i in r:
                ev = Event()
                ev._wrap(self.HashMap[i.decode("UTF-8")])
                ev.ptr = self.ptr.HashToEvent(i)
                out.append(ev)
            return out
        if self.ptr.ContainsHash(i): 
            ev = Event()
            ev._wrap(self.HashMap[key])
            ev.ptr = self.ptr.HashToEvent(<string>key.encode("UTF-8"))
            return ev
        return False

    def __len__(self) -> int:
        if self.ptr.length == 0: self.ptr.HashList()
        return self.ptr.length

    def __add__(self, other) -> SampleTracer:
        if isinstance(self, int): return other
        if isinstance(other, int): return self

        cdef SampleTracer s = self
        cdef SampleTracer o = other
        cdef SampleTracer p = self.clone
        p.ptr = s.ptr[0] + o.ptr
        for i in s.HashMap: p.HashMap[i] = s.HashMap[i] 
        for i in o.HashMap: p.HashMap[i] = o.HashMap[i]
        return p
    
    def __iadd__(self, other):
        cdef SampleTracer s = self
        cdef SampleTracer o = other
        s.ptr = s.ptr[0] + o.ptr
        for i in o.HashMap: s.HashMap[i] = o.HashMap[i]
        return s

    def HashToROOT(self, str key) -> str:
        return self.ptr.HashToROOT(key.encode("UTF-8")).decode("UTF-8")

    @property
    def clone(self):
        v = self.__new__(self.__class__)
        v.__init__()
        return v
    
    @property
    def Threads(self) -> int: return self.ptr.Threads

    @property
    def chnk(self) -> int: return self.ptr.ChunkSize

    @Threads.setter
    def Threads(self, int val): self.ptr.Threads = val

    @chnk.setter
    def chnk(self, int val): self.ptr.ChunkSize = val

    def FastHashSearch(self, hashes) -> dict:
        cdef vector[string] v
        cdef int i
        cdef string key
        cdef bool fnd

        for i in range(len(hashes)): v.push_back(hashes[i].encode("UTF-8"))
        return {key.decode("UTF-8") : fnd for key, fnd in self.ptr.FastSearch(v)}

    def AddEvent(self, Events, root, index):
        cdef int i; 
        for i in range(len(Events)):
            if Events.index == -1: Events.index = index
            Events[i].hash = root
            
            if self.ptr.ContainsHash(Events[i].hash.encode("UTF-8")): continue
            _ev = new CyEvent()
            _ev.Tree = Events[i].Tree.encode("UTF-8")
            _ev.Hash = Events[i].hash.encode("UTF-8")
            _ev.ROOT = root.encode("UTF-8")
            _ev.EventIndex = <int>index
            _ev.Event = True
            self.ptr.AddEvent(_ev) 
            self.HashMap[Events[i].hash] = Events[i]
    
    def AddGraph(self, Events):
        cdef int i;
        cdef str hash; 
        cdef CyEvent* event;
        for i in range(len(Events)):
            hash, evnt = Events[i]
            self.HashMap[hash] = evnt 
            event = self.ptr.HashToEvent(<string>hash.encode("UTF-8"))
            event.Graph = True 

    def __iter__(self):
        self._itv = self.ptr.HashList()
        self._its = 0
        self._ite = self._itv.size()
        return self

    def __next__(self):
        if self._its == self._ite: raise StopIteration
        cdef str hash = self._itv[self._its].decode("UTF-8")
        ev = Event()
        ev._wrap(self.HashMap[hash])
        ev.ptr = self.ptr.HashToEvent(<string>hash.encode("UTF-8"))
        self._its += 1
        return ev
        
