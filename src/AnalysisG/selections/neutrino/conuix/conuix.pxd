# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "conuix.h":
    cdef cppclass conuix(selection_template):
        conuix() except +

cdef class Conuix(SelectionTemplate):
    cdef conuix* tt



