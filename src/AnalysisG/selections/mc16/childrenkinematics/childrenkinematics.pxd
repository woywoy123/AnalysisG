# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "childrenkinematics.h":
    cdef cppclass childrenkinematics(selection_template):
        childrenkinematics() except +

cdef class ChildrenKinematics(SelectionTemplate):
    cdef childrenkinematics* tt

    cdef map[string, vector[float]] r_data
    cdef map[string, vector[float]] s_data
    cdef map[string, vector[float]] r_decay 
    cdef map[string, vector[float]] s_decay
    cdef map[string, vector[float]] top_pem

    cdef public dict res_kinematics
    cdef public dict spec_kinematics
    cdef public dict res_pdgid_kinematics
    cdef public dict spec_pdgid_kinematics

    cdef public dict res_decay_mode
    cdef public dict spec_decay_mode
    cdef public dict fractional

    cdef public dict dr_clustering
    cdef public dict top_children_dr

