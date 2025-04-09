# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topmatching.h":
    cdef cppclass topmatching(selection_template):
        topmatching() except +


cdef class TopMatching(SelectionTemplate):
    cdef topmatching* tt

    cdef vector[float] top_mass
    cdef vector[float] topchildren_mass
    cdef vector[float] toptruthjets_mass
    cdef vector[float] topjets_children_mass
    cdef vector[float] topjets_leptons_mass
    
    cdef vector[bool]  topchildren_islep
    cdef vector[bool]  toptruthjets_islep
    cdef vector[bool]  topjets_children_islep
    cdef vector[bool]  topjets_leptons_islep
    
    cdef vector[int]   toptruthjets_njets
    cdef vector[int]   topjets_leptons_pdgid
    cdef vector[int]   topjets_leptons_njets

    cdef public list truth_top
    cdef public list no_children

    cdef public dict truth_children
    cdef public dict truth_jets
    cdef public dict n_truth_jets_lep
    cdef public dict n_truth_jets_had
    cdef public dict jets_truth_leps
    cdef public dict jet_leps
    cdef public dict n_jets_lep
    cdef public dict n_jets_had



