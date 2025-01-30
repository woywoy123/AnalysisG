# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class ZPrime(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new zprime()
        self.tt = <zprime*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.zprime_truth_tops = self.tt.zprime_truth_tops
        self.zprime_children   = self.tt.zprime_children
        self.zprime_truthjets  = self.tt.zprime_truthjets
        self.zprime_jets       = self.tt.zprime_jets
