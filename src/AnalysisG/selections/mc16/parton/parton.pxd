# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "parton.h":
    cdef cppclass parton(selection_template):
        parton() except +

        map[string, vector[float]] ntops_tjets_pt
        map[string, vector[float]] ntops_tjets_e

        map[string, vector[float]] ntops_jets_pt
        map[string, vector[float]] ntops_jets_e

        map[string, vector[float]] nparton_tjet_e
        map[string, vector[float]] nparton_jet_e

        map[string, vector[float]] frac_parton_tjet_e
        map[string, vector[float]] frac_parton_jet_e

        map[string, vector[float]] frac_ntop_tjet_contribution
        map[string, vector[float]] frac_ntop_jet_contribution

        map[string, vector[float]] frac_mass_top

cdef class Parton(SelectionTemplate):
    cdef parton* tt

    cdef public dict ntops_tjets_pt
    cdef public dict ntops_tjets_e

    cdef public dict ntops_jets_pt
    cdef public dict ntops_jets_e

    cdef public dict nparton_tjet_e
    cdef public dict nparton_jet_e

    cdef public dict frac_parton_tjet_e
    cdef public dict frac_parton_jet_e

    cdef public dict frac_ntop_tjet_contribution
    cdef public dict frac_ntop_jet_contribution

    cdef public dict frac_mass_top
