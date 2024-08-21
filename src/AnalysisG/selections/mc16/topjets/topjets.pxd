# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topjets.h":
    cdef cppclass topjets(selection_template):
        topjets() except +

        map[string, map[string, map[string, vector[float]]]] top_mass
        map[string, map[string, map[string, vector[float]]]] jet_partons
        map[string, map[string, map[string, vector[float]]]] jets_contribute
        map[string, map[string, vector[float]]] jet_top
        map[string, vector[float]] jet_mass
        vector[int] ntops_lost

cdef class TopJets(SelectionTemplate):
    cdef topjets* tt
    cdef public dict top_mass
    cdef public dict jet_partons
    cdef public dict jets_contribute
    cdef public dict jet_top
    cdef public dict jet_mass
    cdef public list ntops_lost

