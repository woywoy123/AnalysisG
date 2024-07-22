# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.performance.topefficiency.topefficiency cimport *

cdef class TopEfficiency(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topefficiency()
        self.tt = <topefficiency*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        self.truthchildren_pt_eta_topmass = inpt["truthchildren_pt_eta_topmass"]
        self.truthjets_pt_eta_topmass     = inpt["truthjets_pt_eta_topmass"]
        self.jets_pt_eta_topmass          = inpt["jets_pt_eta_topmass"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict out = {
            "truthchildren_pt_eta_topmass" : self.truthchildren_pt_eta_topmass,
            "truthjets_pt_eta_topmass    " : self.truthjets_pt_eta_topmass,
            "jets_pt_eta_topmass         " : self.jets_pt_eta_topmass
        }
        return TopEfficiency, (out,)

    cdef void transform_dict_keys(self):
        self.truthchildren_pt_eta_topmass = as_dict(&self.tt.truthchildren_pt_eta_topmass)
        self.truthjets_pt_eta_topmass     = as_dict(&self.tt.truthjets_pt_eta_topmass)
        self.jets_pt_eta_topmass          = as_dict(&self.tt.jets_pt_eta_topmass)


