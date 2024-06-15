# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.graph_template cimport graph_template

from AnalysisG.generators.sampletracer   cimport sampletracer
from AnalysisG.generators.eventgenerator cimport eventgenerator

cdef extern from "<generators/graphgenerator.h>":
    cdef cppclass graphgenerator(sampletracer):

        graphgenerator() except +
        void add_graph_template(map[string, graph_template*]* inpt) except +

cdef struct target_search:
    string event_name
    string graph_name
    string tree

cdef class GraphGenerator:
    cdef graphgenerator* ev_ptr
    cdef vector[eventgenerator*] event_gen
    cdef vector[target_search] targets

    cdef map[string, graph_template*] graph_types
