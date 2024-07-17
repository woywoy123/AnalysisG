# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "toptruthjets.h":
    cdef cppclass toptruthjets(selection_template):
        toptruthjets() except +

cdef class TopTruthJets(SelectionTemplate):
    cdef toptruthjets* tt

