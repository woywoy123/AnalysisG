# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "zprime.h":
    cdef cppclass zprime(selection_template):
        zprime() except +

        vector[float] zprime_pt
        vector[float] zprime_truth_tops
        vector[float] zprime_children
        vector[float] zprime_truthjets
        vector[float] zprime_jets


cdef class ZPrime(SelectionTemplate):
    cdef zprime* tt

    cdef public list zprime_truth_tops
    cdef public list zprime_children
    cdef public list zprime_truthjets
    cdef public list zprime_jets
