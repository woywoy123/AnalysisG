# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.events.gnn.particle_gnn cimport *
from AnalysisG.core.event_template cimport *

cdef extern from "<inference/gnn-event.h>":

    cdef cppclass gnn_event(event_template):
        gnn_event() except+

cdef class EventGNN(EventTemplate):
    cdef gnn_event* ev
