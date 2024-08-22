# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class ZPrime(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new zprime()
        self.tt = <zprime*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        self.zprime_truth_tops = inpt["zprime_truth_tops"]
        self.zprime_children   = inpt["zprime_children"]
        self.zprime_truthjets  = inpt["zprime_truthjets"]
        self.zprime_jets       = inpt["zprime_jets"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict dct = {
                "zprime_truth_tops" : self.zprime_truth_tops,
                "zprime_children"   : self.zprime_children,
                "zprime_truthjets"  : self.zprime_truthjets,
                "zprime_jets"       : self.zprime_jets
        }
        return ZPrime, (dct, )

    cdef void transform_dict_keys(self):
        self.zprime_truth_tops = self.tt.zprime_truth_tops
        self.zprime_children   = self.tt.zprime_children
        self.zprime_truthjets  = self.tt.zprime_truthjets
        self.zprime_jets       = self.tt.zprime_jets
