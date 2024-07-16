# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "decaymodes.h":
    cdef cppclass decaymodes(selection_template):
        decaymodes() except +



cdef class DecayModes(SelectionTemplate):
    cdef decaymodes* tt



