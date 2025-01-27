# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef class DecayModes(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new decaymodes()
        self.tt = <decaymodes*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.res_top_modes     = as_dict(&self.tt.res_top_modes)
        self.res_top_charges   = as_dict(&self.tt.res_top_charges)
        self.res_top_pdgid     = as_basic_dict(&self.tt.res_top_pdgid)
        self.spec_top_modes    = as_dict(&self.tt.spec_top_modes)
        self.spec_top_charges  = as_dict(&self.tt.spec_top_charges)
        self.spec_top_pdgid    = as_basic_dict(&self.tt.spec_top_pdgid)
        self.all_pdgid         = as_basic_dict(&self.tt.all_pdgid)
        self.signal_region     = as_dict(&self.tt.signal_region)
        self.lepton_statistics = as_basic_dict(&self.tt.lepton_statistics)
        self.ntops             = list(self.tt.ntops)
