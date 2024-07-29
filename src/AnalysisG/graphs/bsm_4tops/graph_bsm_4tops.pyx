# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport GraphTemplate
from AnalysisG.graphs.bsm_4tops.graph_bsm_4tops cimport *

cdef class GraphTops(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_tops()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphChildren(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_children()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphTruthJets(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_truthjets()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class GraphTruthJetsNoNu(GraphTemplate):

    def __cinit__(self): self.ptr = new graph_truthjets_nonu()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

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

