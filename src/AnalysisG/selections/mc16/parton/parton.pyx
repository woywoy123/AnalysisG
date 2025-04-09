# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class Parton(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new parton()
        self.tt = <parton*>self.ptr

    def __dealloc__(self): del self.ptr

    cdef void transform_dict_keys(self):
        self.ntops_tjets_pt               = as_dict(&self.tt.ntops_tjets_pt)
        self.ntops_tjets_e                = as_dict(&self.tt.ntops_tjets_e)
        self.ntops_jets_pt                = as_dict(&self.tt.ntops_jets_pt)
        self.ntops_jets_e                 = as_dict(&self.tt.ntops_jets_e)
        self.nparton_tjet_e               = as_dict(&self.tt.nparton_tjet_e)
        self.nparton_jet_e                = as_dict(&self.tt.nparton_jet_e)
        self.frac_parton_tjet_e           = as_dict(&self.tt.frac_parton_tjet_e)
        self.frac_parton_jet_e            = as_dict(&self.tt.frac_parton_jet_e)
        self.frac_ntop_tjet_contribution  = as_dict(&self.tt.frac_ntop_tjet_contribution)
        self.frac_ntop_jet_contribution   = as_dict(&self.tt.frac_ntop_jet_contribution)
        self.frac_mass_top                = as_dict(&self.tt.frac_mass_top)
