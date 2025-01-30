# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topkinematics.h":
    cdef cppclass topkinematics(selection_template):
        topkinematics() except +

        map[string, vector[float]] res_top_kinematics
        map[string, vector[float]] spec_top_kinematics
        map[string, vector[float]] mass_combi
        map[string, vector[float]] deltaR

cdef class TopKinematics(SelectionTemplate):
    cdef topkinematics* tt
    cdef public dict res_top_kinematics
    cdef public dict spec_top_kinematics
    cdef public dict mass_combi
    cdef public dict deltaR
