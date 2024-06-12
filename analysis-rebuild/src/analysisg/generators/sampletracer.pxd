# distutils: langauge = c++
# cython: language_level = 3

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from analysisg.core.event_template cimport event_template
from analysisg.core.graph_template cimport graph_template

cdef extern from "<generators/sampletracer.h>":

    cdef cppclass container:
        container() except+
        int threads

    cdef cppclass sampletracer:

        sampletracer() except+
        vector[event_template*] get_event(string type_, string tree_) except+
        vector[graph_template*] get_graph(string type_, string tree_) except+

        map[string, container*]* root_container
