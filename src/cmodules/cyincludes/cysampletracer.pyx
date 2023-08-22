# distuils: language = c++
# cython: language_level = 3

from cyevent cimport CyEventTemplate, ExportEventTemplate
from cymetadata cimport CyMetaData, ExportMetaData
from cysampletracer cimport CySampleTracer
from AnalysisG.Tools import Code
from cycode cimport ExportCode, CyCode
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair
from typing import Union

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class SampleTracer:

    cdef CySampleTracer* ptr

    def __cinit__(self):
        self.ptr = new CySampleTracer()

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
        return max([it.second for it in f])

    def AddEvent(self, event, meta):
        cdef ExportMetaData m = meta.__getstate__()
        cdef ExportEventTemplate ev = event.__getstate__()
        code = Code(event)
        cdef dict Objects = event.Objects
        for obj in Objects.values():
            code.add_dependency(Code(obj).__getstate__())
        self.ptr.AddEvent(ev, m, code)

    def __preiteration__(self) -> bool:
        return False

    def __iter__(self):
        return self

    def __next__(self):
        return self

