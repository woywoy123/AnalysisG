# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.string cimport string

from AnalysisG.core.io cimport IO
from AnalysisG.core.event_template cimport event_template
from AnalysisG.generators.sampletracer cimport sampletracer

cdef extern from "<generators/eventgenerator.h>":
    cdef cppclass eventgenerator(sampletracer):

        eventgenerator() except +
        void add_event_template(map[string, event_template*]* inpt) except +
        void flush_events() except+

cdef class EventGenerator(IO):
    cdef eventgenerator* ev_ptr
    cdef map[string, event_template*] event_types
    cdef void event_compiler(self, event_template*)
    cdef void flush(self)
