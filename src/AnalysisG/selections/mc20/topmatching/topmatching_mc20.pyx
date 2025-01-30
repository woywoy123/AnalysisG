# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class TopMatching(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topmatching()
        self.tt = <topmatching*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.truth_top        = as_list(&self.tt.truth_top)
        self.no_children      = as_list(&self.tt.no_children)

        self.truth_children   = as_dict(&self.tt.truth_children)
        self.truth_jets       = as_dict(&self.tt.truth_jets)
        self.n_truth_jets_lep = as_dict(&self.tt.n_truth_jets_lep)
        self.n_truth_jets_had = as_dict(&self.tt.n_truth_jets_had)
        self.jets_truth_leps  = as_dict(&self.tt.jets_truth_leps)
        self.jet_leps         = as_dict(&self.tt.jet_leps)
        self.n_jets_lep       = as_dict(&self.tt.n_jets_lep)
        self.n_jets_had       = as_dict(&self.tt.n_jets_had)





