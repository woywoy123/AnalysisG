# distutils: language = c++
# cython: language_level = 3

from libcpp.map cimport map
from libcpp.string cimport string

from analysisg.generators.sampletracer   cimport sampletracer
from analysisg.generators.eventgenerator cimport eventgenerator

from analysisg.core.event_template cimport event_template
from analysisg.core.graph_template cimport graph_template

cdef extern from "<generators/graphgenerator.h>":
    cdef cppclass graphgenerator(sampletracer):

        graphgenerator() except +
        void add_graph_template(map[string, graph_template*]* inpt) except +

cdef class GraphGenerator:
    cdef graphgenerator* ev_ptr
    cdef map[string, graph_template*] event_types
    cdef void graph_compiler(self, graph_template*, event_template*)
