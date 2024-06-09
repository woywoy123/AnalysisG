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
        self.__iter__()
        cdef map[string, event_template*] ev
        cdef bool run = True
        cdef int x = 0
        while run:
            ev = p.build_event(deref(self.data_ops))
            run = ev.size()

            x += 1
            if x%1000 == 1: print(x)


    def ImportEvent(self, evn):
        cdef EventTemplate ev = evn
        cdef event_template* data = ev.ptr

        self.ptr.leaves = data.leaves
        self.ptr.branches = data.branches
        self.ptr.trees = data.trees
        if not self.ptr.scan_keys(): exit()
        self.event_compiler(data)
