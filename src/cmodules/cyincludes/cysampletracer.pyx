# distuils: language = c++
# cython: language_level = 3

from cytypes cimport meta_t, event_T, event_t, code_t
from cysampletracer cimport CySampleTracer, CyBatch

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair

from AnalysisG.Tools import Code
from typing import Union

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Event:

    cdef CyBatch* ptr

    def __cinit__(self):
        self.ptr = NULL

    @property
    def hash(self): return env(self.ptr.Hash())





cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef dict hashed_code
    cdef vector[CyBatch*] itb
    cdef unsigned int its
    cdef unsigned int ite

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self.hashed_code = {}

    def __dealloc__(self):
        del self.ptr

    def __init__(self):
        pass

    def __getitem__(self, key: Union[list, str]):
        cdef vector[string] inpt;
        cdef str it
        if isinstance(key, str): inpt = [enc(key)]
        else: inpt = [enc(it) for it in key]
        #return self.ptr.Search(inpt)

    def __len__(self) -> int:
        cdef map[string, int] f = self.ptr.length()
        cdef pair[string, int] it
        cdef list entries = [it.second for it in f]
        return max(entries) if len(entries) else 0

    def AddEvent(self, event, meta):
        cdef meta_t m = meta.__getstate__()
        cdef event_T ev = event.__getstate__()

        cdef code_t it
        cdef vector[code_t] code = []
        cdef dict Objects = event.Objects
        cdef str event_name = env(ev.event.event_name)
        if event_name not in self.hashed_code:
            for o in list(Objects.values()) + [event]:
                it = Code(o).__getstate__()
                code.push_back(it)
            self.hashed_code[event_name] = True
        self.ptr.AddEvent(ev.event, m, code)

    def __preiteration__(self) -> bool:
        return False

    def __iter__(self):
        if self.__preiteration__(): return self
        self.itb = self.ptr.MakeIterable()
        self.ite = self.itb.size()
        self.its = 0
        return self

    def __next__(self) -> Event:
        if self.its == self.ite: raise StopIteration
        cdef CyBatch* b = self.itb[self.its]
        cdef Event ev = Event()
        ev.ptr = b
        self.its+=1
        return ev

