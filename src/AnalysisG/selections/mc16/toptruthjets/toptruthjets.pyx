# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.mc16.toptruthjets.toptruthjets cimport *
from AnalysisG.core.tools cimport *

cdef class TopTruthJets(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new toptruthjets()
        self.tt = <toptruthjets*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        self.top_mass             = inpt["top_mass"]
        self.truthjet_partons     = inpt["truthjet_partons"]
        self.truthjets_contribute = inpt["truthjets_contribute"]
        self.truthjet_top         = inpt["truthjet_top"]
        self.truthjet_mass        = inpt["truthjet_mass"]
        self.ntops_lost           = inpt["ntops_lost"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict out = {
                "top_mass" :  self.top_mass,
                "truthjet_partons" :  self.truthjet_partons,
                "truthjets_contribute" :  self.truthjets_contribute,
                "truthjet_top" :  self.truthjet_top,
                "truthjet_mass" :  self.truthjet_mass,
                "ntops_lost" :  self.ntops_lost
        }
        return TopTruthJets, (out,)

    cdef void transform_dict_keys(self):
        self.top_mass             = as_dict_dict_dict(&self.tt.top_mass)
        self.truthjet_partons     = as_dict_dict_dict(&self.tt.truthjet_partons)
        self.truthjets_contribute = as_dict_dict_dict(&self.tt.truthjets_contribute)
        self.truthjet_top         = as_dict_dict(&self.tt.truthjet_top)
        self.truthjet_mass        = as_dict(&self.tt.truthjet_mass)
        self.ntops_lost           = as_list(&self.tt.ntops_lost)

