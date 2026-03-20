# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "multisol.h":
    cdef cppclass multisol(selection_template):
        multisol() except +

cdef class MultiSol(SelectionTemplate):
    cdef multisol* tt



