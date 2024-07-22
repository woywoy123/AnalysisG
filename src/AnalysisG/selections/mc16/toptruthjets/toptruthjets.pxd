# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "toptruthjets.h":
    cdef cppclass toptruthjets(selection_template):
        toptruthjets() except +

        map[string, map[string, map[string, vector[float]]]] top_mass
        map[string, map[string, map[string, vector[float]]]] truthjet_partons
        map[string, map[string, map[string, vector[float]]]] truthjets_contribute
        map[string, map[string, vector[float]]] truthjet_top
        map[string, vector[float]] truthjet_mass
        vector[int] ntops_lost

cdef class TopTruthJets(SelectionTemplate):
    cdef toptruthjets* tt
    cdef public dict top_mass
    cdef public dict truthjet_partons
    cdef public dict truthjets_contribute
    cdef public dict truthjet_top
    cdef public dict truthjet_mass
    cdef public list ntops_lost

