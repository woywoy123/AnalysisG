# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "met.h":
    cdef cppclass met(selection_template):
        met() except +
        map[string, float] missing_et

cdef class MET(SelectionTemplate):
    cdef met* tt
    cdef public dict missing_et



