# distutils: language=c++
# cython: language_level=3

from tqdm import tqdm
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map, pair

from AnalysisG.core.meta cimport meta
from AnalysisG.core.io cimport IO, data_t
from AnalysisG.core.event_template cimport EventTemplate, event_template
from AnalysisG.generators.eventgenerator cimport eventgenerator

cdef class EventGenerator(IO):

    def __cinit__(self): self.ev_ptr = new eventgenerator()
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr

    cdef void flush(self): self.ev_ptr.flush_events()
    cdef void event_compiler(self, event_template* p):

        self.ptr.leaves   = p.leaves
        self.ptr.branches = p.branches
        self.ptr.trees    = p.trees
        if not self.ptr.scan_keys(): exit()
        self.__iter__()

        cdef int i
        cdef map[string, event_template*] ev

        cdef int l = len(self)
        for i in tqdm(range(l)):
            ev = p.build_event(self.data_ops)
            if not ev.size(): continue
            self.ev_ptr.add_event_template(&ev)
        self.ev_ptr.compile()

    def ImportEvent(self, evn):
        cdef EventTemplate ev = evn
        cdef event_template* data = ev.ptr
        self.event_types[data.name] = data

    def CompileEvents(self):
        cdef pair[string, event_template*] itr
        for itr in self.event_types: self.event_compiler(itr.second)
