# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef class DecayModes(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new decaymodes()
        self.tt = <decaymodes*>self.ptr

    def __init__(self, data = None):
        if data is None: return
        self.res_top_modes    = data["res_top_modes"]
        self.res_top_charges  = data["res_top_charges"]
        self.res_top_pdgid    = data["res_top_pdgid"]
        self.spec_top_modes   = data["spec_top_modes"]
        self.spec_top_charges = data["spec_top_charges"]
        self.spec_top_pdgid   = data["spec_top_pdgid"]
        self.all_pdgid        = data["all_pdgid"]
        self.signal_region    = data["signal_region"]
        self.ntops            = data["ntops"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict tmp = {
            "res_top_modes"    :  self.res_top_modes   ,
            "res_top_charges"  :  self.res_top_charges ,
            "res_top_pdgid"    :  self.res_top_pdgid   ,
            "spec_top_modes"   :  self.spec_top_modes  ,
            "spec_top_charges" :  self.spec_top_charges,
            "spec_top_pdgid"   :  self.spec_top_pdgid  ,
            "all_pdgid"        :  self.all_pdgid       ,
            "signal_region"    :  self.signal_region   ,
            "ntops"            :  self.ntops
        }
        return DecayModes, (tmp,)

    cdef void transform_dict_keys(self):
        self.res_top_modes    = as_dict(&self.tt.res_top_modes)
        self.res_top_charges  = as_dict(&self.tt.res_top_charges)
        self.res_top_pdgid    = as_basic_dict(&self.tt.res_top_pdgid)
        self.spec_top_modes   = as_dict(&self.tt.spec_top_modes)
        self.spec_top_charges = as_dict(&self.tt.spec_top_charges)
        self.spec_top_pdgid   = as_basic_dict(&self.tt.spec_top_pdgid)
        self.all_pdgid        = as_basic_dict(&self.tt.all_pdgid)
        self.signal_region    = as_dict(&self.tt.signal_region)
        self.ntops            = list(self.tt.ntops)
