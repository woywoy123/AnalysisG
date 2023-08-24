# distuils: language = c++
# cython: language_level = 3

from cysampletracer cimport CySampleTracer
from cyevent cimport CyEventTemplate
from cymetadata cimport CyMetaData
from cycode cimport CyCode
from cytypes cimport *

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair

from AnalysisG.Tools import Code
from typing import Union

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef dict hashed_code
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
        return self.ptr.Search(inpt)

    def __len__(self) -> int:
        cdef map[string, unsigned int] f = self.ptr.Length()
        cdef pair[string, unsigned int] it
        cdef list entries = [it.second for it in f]
        return max(entries) if len(entries) else 0

    def AddEvent(self, event, meta):
        cdef ExportMetaData m = meta.__getstate__()
        cdef ExportEventTemplate ev = event.__getstate__()

        cdef ExportCode it
        cdef vector[ExportCode] code = []
        cdef dict Objects = event.Objects
        cdef str event_name = env(ev.event_name)
        if event_name not in self.hashed_code:
            for o in list(Objects.values()) + [event]:
                it = Code(o).__getstate__()
                code.push_back(it)
            self.hashed_code[event_name] = True
        self.ptr.AddEvent(ev, m, code)

    def __preiteration__(self) -> bool:
        return False

    def __iter__(self):
        if self.__preiteration__(): return self
        return self

    def __next__(self):
        return self

