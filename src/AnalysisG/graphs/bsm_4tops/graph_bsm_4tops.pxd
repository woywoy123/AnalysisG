# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.graph_template cimport graph_template, GraphTemplate

cdef extern from "<bsm_4tops/graphs.h>":
    cdef cppclass graph_tops(graph_template):
        graph_tops() except+

    cdef cppclass graph_children(graph_template):
        graph_children() except+

    cdef cppclass graph_truthjets(graph_template):
        graph_truthjets() except+

    cdef cppclass graph_truthjets_nonu(graph_template):
        graph_truthjets_nonu() except+

    cdef cppclass graph_jets(graph_template):
        graph_jets() except+

    cdef cppclass graph_jets_nonu(graph_template):
        graph_jets_nonu() except+

    cdef cppclass graph_jets_detector_lep(graph_template):
        graph_jets_detector_lep() except+

    cdef cppclass graph_detector(graph_template):
        graph_detector() except+


cdef class GraphTops(GraphTemplate):
    pass

cdef class GraphChildren(GraphTemplate):
    pass

cdef class GraphTruthJets(GraphTemplate):
    pass

cdef class GraphTruthJetsNoNu(GraphTemplate):
    pass

cdef class GraphJets(GraphTemplate):
    pass

cdef class GraphJetsNoNu(GraphTemplate):
    pass

cdef class GraphDetectorLep(GraphTemplate):
    pass

cdef class GraphDetector(GraphTemplate):
    pass

