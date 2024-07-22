# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topefficiency.h":
    cdef cppclass topefficiency(selection_template):
        topefficiency() except +

        map[string, vector[float]] truthchildren_pt_eta_topmass
        map[string, vector[float]] truthjets_pt_eta_topmass
        map[string, vector[float]] jets_pt_eta_topmass

cdef class TopEfficiency(SelectionTemplate):
    cdef topefficiency* tt

    cdef public dict truthchildren_pt_eta_topmass
    cdef public dict truthjets_pt_eta_topmass
    cdef public dict jets_pt_eta_topmass
