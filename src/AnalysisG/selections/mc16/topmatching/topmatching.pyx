# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class TopMatching(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topmatching()
        self.tt = <topmatching*>self.ptr

    def __init__(self, 
            v1 = None, v2 = None, v3 = None, v4 = None,  v5 = None,
            v6 = None, v7 = None, v8 = None, v9 = None, v10 = None
        ):
        if v1 is None: return
        self.truth_top        = v1
        self.no_children      = v2
        self.truth_children   = v3
        self.truth_jets       = v4
        self.n_truth_jets_lep = v5
        self.n_truth_jets_had = v6
        self.jets_truth_leps  = v7
        self.jet_leps        = v8
        self.n_jets_lep       = v9
        self.n_jets_had       = v10

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        out = (
            self.truth_top,
            self.no_children,
            self.truth_children,
            self.truth_jets,
            self.n_truth_jets_lep,
            self.n_truth_jets_had,
            self.jets_truth_leps,
            self.jet_leps,
            self.n_jets_lep,
            self.n_jets_had,
        )
        return TopMatching, out

    cdef void transform_dict_keys(self):
        self.truth_top        = as_list(&self.tt.truth_top)
        self.no_children      = as_list(&self.tt.no_children)

        self.truth_children   = as_dict(&self.tt.truth_children)
        self.truth_jets       = as_dict(&self.tt.truth_jets)
        self.n_truth_jets_lep = as_dict(&self.tt.n_truth_jets_lep)
        self.n_truth_jets_had = as_dict(&self.tt.n_truth_jets_had)
        self.jets_truth_leps  = as_dict(&self.tt.jets_truth_leps)
        self.jet_leps        = as_dict(&self.tt.jet_leps)
        self.n_jets_lep       = as_dict(&self.tt.n_jets_lep)
        self.n_jets_had       = as_dict(&self.tt.n_jets_had)





