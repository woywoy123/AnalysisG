# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport GraphTemplate
from AnalysisG.graphs.bsm_4tops.graph_bsm_4tops cimport *
from libcpp cimport bool, int

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

    def __cinit__(self): 
        self.ptr = new graph_detector()
        self.tt = <graph_detector*>(self.ptr)
    def __init__(self): pass
    def __dealloc__(self): del self.tt

    @property
    def NumCuda(self): return self.tt.num_cuda
    @NumCuda.setter
    def NumCuda(self, int v): self.tt.num_cuda = v

    @property
    def ForceMatch(self): return self.tt.force_match
    @ForceMatch.setter
    def ForceMatch(self, bool v): self.tt.force_match = v



