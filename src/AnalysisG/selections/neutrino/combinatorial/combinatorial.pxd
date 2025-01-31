# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "combinatorial.h":
    cdef cppclass combinatorial(selection_template):
        combinatorial() except +
        map[string, double] delta_met
        map[string, double] delta_metnu
        map[string, double] obs_met
        map[string, double] nus_met
        map[string, double] dist_nu

        map[string, vector[int]] pdgid
        map[string, vector[double]] tru_topmass
        map[string, vector[double]] tru_wmass

        map[string, vector[double]] exp_topmass
        map[string, vector[double]] exp_wmass

cdef class Combinatorial(SelectionTemplate):
    cdef combinatorial* tt

    cdef public dict delta_met
    cdef public dict delta_metnu
    cdef public dict obs_met
    cdef public dict nus_met

    cdef public dict dist_nu
    cdef public dict pdgid

    cdef public dict tru_topmass
    cdef public dict tru_wmass

    cdef public dict exp_topmass
    cdef public dict exp_wmass
