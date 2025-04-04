# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

def fx(zp, data):
    name, vx = data
    val = getattr(zp, name)
    if val is not None: setattr(zp, name, val + vx)
    else: setattr(zp, name, []); fx(zp, data)

cdef class ZPrime(SelectionTemplate):
    def __cinit__(self):
        self.root_leaves = {
            "zprime_truth_tops": fx, 
            "zprime_children"  : fx,
            "zprime_truthjets" : fx,
            "zprime_jets"      : fx,
        }
        self.ptr = new zprime()
        self.tt = <zprime*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.zprime_truth_tops = self.tt.zprime_truth_tops
        self.zprime_children   = self.tt.zprime_children
        self.zprime_truthjets  = self.tt.zprime_truthjets
        self.zprime_jets       = self.tt.zprime_jets
