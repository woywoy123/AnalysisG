# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from analysisg.core.graph_template cimport graph_template, GraphTemplate

from analysisg.generators.graphgenerator cimport graphgenerator
from analysisg.generators.eventgenerator cimport eventgenerator, EventGenerator

cdef class GraphGenerator:
    def __cinit__(self): self.ev_ptr = new graphgenerator()
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr

    cdef void graph_compiler(self, graph_template* p, event_template* ev):
        ev.CompileEvent()
        p.build_event(ev)


    def ImportGraph(self, GraphTemplate evn):
        cdef graph_template* data = evn.ptr
        self.event_types[data.name] = data

    def CompileEvents(self):
        cdef pair[string, graph_template*] itr

    def AddEvents(self, EventGenerator evg, str sample_name = ""):
        cdef eventgenerator* eg = evg.ev_ptr
        cdef vector[event_template*] evs = eg.get_event(b"", b"")

        cdef pair[string, graph_template*] itr
        for itr in self.event_types: self.graph_compiler(itr.second, evs[0])











