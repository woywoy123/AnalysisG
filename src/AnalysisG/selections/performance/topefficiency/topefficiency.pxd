# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topefficiency.h":
    cdef cppclass topefficiency(selection_template):
        topefficiency() except +

cdef class TopEfficiency(SelectionTemplate):
    cdef topefficiency* tt



