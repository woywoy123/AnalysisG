# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map, pair
from cython.operator cimport dereference as deref

from analysisg.generators.eventgenerator cimport eventgenerator
from analysisg.core.event_template cimport EventTemplate, event_template
from analysisg.core.meta cimport meta
from analysisg.core.io cimport IO, data_t

cdef class EventGenerator(IO):

    def __cinit__(self): self.ev_ptr = new eventgenerator()
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr

    cdef void event_compiler(self, event_template* p):

        self.ptr.leaves   = p.leaves
        self.ptr.branches = p.branches
        self.ptr.trees    = p.trees
        if not self.ptr.scan_keys(): exit()
        self.__iter__()

        cdef int i
        cdef map[string, event_template*] ev
        for i in range(len(self)):
            ev = p.build_event(deref(self.data_ops))
            self.ev_ptr.add_event_template(&ev)
            break

    def ImportEvent(self, evn):
        cdef EventTemplate ev = evn
        cdef event_template* data = ev.ptr
        self.event_types[data.name] = data

    def CompileEvents(self):
        cdef pair[string, event_template*] itr
        for itr in self.event_types: self.event_compiler(itr.second)
