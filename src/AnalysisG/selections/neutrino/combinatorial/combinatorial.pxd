# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "combinatorial.h":
    cdef cppclass combinatorial(selection_template):
        combinatorial() except +
        int num_device
        double masstop
        double massw

cdef cppclass neutrino(particle_template):
    neutrino() except+ nogil
    double ellipse 
    double chi2

    int matched_bquark
    int matched_lepton
     

cdef class NuNuCombinatorial(SelectionTemplate):
    cdef combinatorial* tt

    cdef int ix
    cdef int lx

    cdef map[string, vector[vector[vector[double]]]] pmu
    cdef map[string, vector[vector[int]]] pdgid

    cdef map[string, vector[vector[int]]] matched_bq
    cdef map[string, vector[vector[int]]] matched_lp
    
    cdef map[string, vector[vector[double]]] ellipse 
    cdef map[string, vector[vector[double]]] chi2_nu1
    cdef map[string, vector[vector[double]]] chi2_nu2
    cdef map[string, vector[vector[vector[double]]]] pmu_nu1
    cdef map[string, vector[vector[vector[double]]]] pmu_nu2

