# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.mc16.topjets.topjets cimport *
from AnalysisG.core.tools cimport *

cdef class TopJets(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topjets()
        self.tt = <topjets*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        self.top_mass        = inpt["top_mass"]
        self.jet_partons     = inpt["jet_partons"]
        self.jets_contribute = inpt["jets_contribute"]
        self.jet_top         = inpt["jet_top"]
        self.jet_mass        = inpt["jet_mass"]
        self.ntops_lost      = inpt["ntops_lost"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict out = {
                "top_mass"        :  self.top_mass,
                "jet_partons"     :  self.jet_partons,
                "jets_contribute" :  self.jets_contribute,
                "jet_top"         :  self.jet_top,
                "jet_mass"        :  self.jet_mass,
                "ntops_lost"      :  self.ntops_lost
        }
        return TopJets, (out,)

    cdef void transform_dict_keys(self):
        self.top_mass        = as_dict_dict_dict(&self.tt.top_mass)
        self.jet_partons     = as_dict_dict_dict(&self.tt.jet_partons)
        self.jets_contribute = as_dict_dict_dict(&self.tt.jets_contribute)
        self.jet_top         = as_dict_dict(&self.tt.jet_top)
        self.jet_mass        = as_dict(&self.tt.jet_mass)
        self.ntops_lost      = as_list(&self.tt.ntops_lost)

