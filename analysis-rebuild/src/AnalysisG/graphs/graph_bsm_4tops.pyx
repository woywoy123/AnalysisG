# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport GraphTemplate

cdef class TruthTops(GraphTemplate):

    def __cinit__(self): self.ptr = new truth_tops()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr


