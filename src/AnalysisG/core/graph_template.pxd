# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from AnalysisG.core.event_template cimport event_template

cdef extern from "<templates/graph_template.h>" nogil:
    cdef cppclass graph_template:
        graph_template() except + nogil

        string tree
        string name
        string hash
        double index

        graph_template* build(event_template*) except+ nogil
        bool operator == (graph_template& p) except+ nogil

cdef class GraphTemplate:
    cdef graph_template* ptr

