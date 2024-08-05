# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.event_template cimport EventTemplate
from AnalysisG.events.gnn.event_gnn cimport *

cdef class EventGNN(EventTemplate):

    def __cinit__(self):
        self.ev = new gnn_event()
        self.ptr = <event_template*>(self.ev)

    def __init__(self): pass
    def __dealloc__(self): del self.ev

