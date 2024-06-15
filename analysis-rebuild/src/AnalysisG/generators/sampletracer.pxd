# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.graph_template cimport graph_template

cdef extern from "<generators/sampletracer.h>":

    cdef cppclass container:
        container() except +
        void flush_events() except +
        void register_event(map[string, event_template*]* inpt) except +
        void register_event(map[string, graph_template*]* inpt) except +
        int threads

    cdef cppclass sampletracer:

        sampletracer() except+

        void compile() except+
        vector[event_template*] get_event(string type_, string tree_) except+
        vector[graph_template*] get_graph(string type_, string tree_) except+
        vector[graph_template*]* delegate_data() except+
        map[string, container*]* root_container
