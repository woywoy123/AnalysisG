# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport graph_template, GraphTemplate

cdef extern from "bsm_4tops/graphs.h":
    cdef cppclass truth_tops(graph_template):
        truth_tops() except+


cdef class TruthTops(GraphTemplate):
    pass
