# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.mc16.toptruthjets.toptruthjets cimport *
from AnalysisG.core.tools cimport *

cdef class TopTruthJets(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new toptruthjets()
        self.tt = <toptruthjets*>self.ptr

    def __dealloc__(self): del self.ptr

    cdef void transform_dict_keys(self):
        self.top_mass             = as_dict_dict_dict(&self.tt.top_mass)
        self.truthjet_partons     = as_dict_dict_dict(&self.tt.truthjet_partons)
        self.truthjets_contribute = as_dict_dict_dict(&self.tt.truthjets_contribute)
        self.truthjet_top         = as_dict_dict(&self.tt.truthjet_top)
        self.truthjet_mass        = as_dict(&self.tt.truthjet_mass)
        self.ntops_lost           = as_list(&self.tt.ntops_lost)

