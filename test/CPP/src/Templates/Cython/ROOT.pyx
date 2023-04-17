#distutils: language = c++
from ROOT cimport CySampleTracer, CyROOT, CyEvent
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

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
            for i in r: out.append(self.HashMap[i]) 
            return out
        if self.ptr.ContainsHash(i): 
            return self.HashMap[key]
        return False

    def __len__(self) -> int:
        if self.ptr.length == 0: self.ptr.HashList()
        return self.ptr.length

    def HashToROOT(self, str key) -> str:
        return self.ptr.HashToROOT(key.encode("UTF-8")).decode("UTF-8")

    def FastHashSearch(self, hashes) -> dict:
        cdef vector[string] v
        cdef int i
        cdef string key
        cdef bool fnd

        for i in range(len(hashes)): v.push_back(hashes[i].encode("UTF-8"))
        return {key.decode("UTF-8") : fnd for key, fnd in self.ptr.FastSearch(v)}


    def AddEvent(self, Events, root, index) -> void:
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

    def __iter__(self):
        self._itv = self.ptr.HashList()
        self._its = 0
        self._ite = self._itv.size()
        return self

    def __next__(self):
        if self._its == self._ite: raise StopIteration
        v = self.HashMap[self._itv[self._its].decode("UTF-8")]
        self._its += 1
        return v
        
