# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport GraphTemplate

cdef class GraphTops(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_tops()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphChildren(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_children()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr


