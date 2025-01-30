# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topmatching.h":
    cdef cppclass topmatching(selection_template):
        topmatching() except +

        vector[float] truth_top
        vector[int] no_children
        map[string, vector[float]] truth_children
        map[string, vector[float]] truth_jets

        map[string, vector[float]] n_truth_jets_lep
        map[string, vector[float]] n_truth_jets_had

        map[string, vector[float]] jets_truth_leps
        map[string, vector[float]] jet_leps

        map[string, vector[float]] n_jets_lep
        map[string, vector[float]] n_jets_had

cdef class TopMatching(SelectionTemplate):
    cdef topmatching* tt

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



