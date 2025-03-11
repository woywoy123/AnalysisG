# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport GraphTemplate
from AnalysisG.graphs.exp_mc20.graph_exp_mc20 cimport *

cdef class GraphJets(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_jets()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphJetsNoNu(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_jets_nonu()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr


cdef class GraphDetectorLep(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_jets_detector_lep()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphDetector(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_detector()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

