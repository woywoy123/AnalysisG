# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "decaymodes.h":
    cdef cppclass decaymodes(selection_template):
        decaymodes() except +
        map[string, vector[double]] res_top_modes
        map[string, vector[double]] res_top_charges
        map[string, int] res_top_pdgid

        map[string, vector[double]] spec_top_modes
        map[string, vector[double]] spec_top_charges
        map[string, int] spec_top_pdgid

        map[string, int] all_pdgid
        map[string, vector[double]] signal_region
        vector[int] ntops


cdef class DecayModes(SelectionTemplate):
    cdef decaymodes* tt
    cdef public dict res_top_modes
    cdef public dict res_top_charges
    cdef public dict res_top_pdgid

    cdef public dict spec_top_modes
    cdef public dict spec_top_charges
    cdef public dict spec_top_pdgid

    cdef public dict all_pdgid
    cdef public dict signal_region
    cdef public list ntops



