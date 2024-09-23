# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.mc16.topjets.topjets cimport *
from AnalysisG.core.tools cimport *

cdef class TopJets(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topjets()
        self.tt = <topjets*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.top_mass        = as_dict_dict_dict(&self.tt.top_mass)
        self.jet_partons     = as_dict_dict_dict(&self.tt.jet_partons)
        self.jets_contribute = as_dict_dict_dict(&self.tt.jets_contribute)
        self.jet_top         = as_dict_dict(&self.tt.jet_top)
        self.jet_mass        = as_dict(&self.tt.jet_mass)
        self.ntops_lost      = as_list(&self.tt.ntops_lost)

