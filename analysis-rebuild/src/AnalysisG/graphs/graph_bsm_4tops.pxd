# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport graph_template, GraphTemplate

cdef extern from "bsm_4tops/graphs.h":
    cdef cppclass graph_tops(graph_template):
        graph_tops() except+

    cdef cppclass graph_children(graph_template):
        graph_children() except+


cdef class GraphTops(GraphTemplate):
    pass

cdef class GraphChildren(GraphTemplate):
    pass
