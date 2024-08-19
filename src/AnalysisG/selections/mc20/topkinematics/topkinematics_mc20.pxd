# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topkinematics.h":
    cdef cppclass topkinematics(selection_template):
        topkinematics() except +

cdef class TopKinematics(SelectionTemplate):
    cdef topkinematics* tt



