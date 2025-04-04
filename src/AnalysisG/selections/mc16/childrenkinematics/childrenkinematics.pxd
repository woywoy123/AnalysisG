# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "childrenkinematics.h":
    cdef cppclass childrenkinematics(selection_template):
        childrenkinematics() except +

#        map[string, vector[float]] res_kinematics
#        map[string, vector[float]] spec_kinematics
#
#        map[string, map[string, vector[float]]] res_pdgid_kinematics
#        map[string, map[string, vector[float]]] spec_pdgid_kinematics
#
#        map[string, map[string, vector[float]]] res_decay_mode
#        map[string, map[string, vector[float]]] spec_decay_mode
#
#        map[string, vector[float]] mass_clustering
#        map[string, vector[float]] dr_clustering
#        map[string, vector[float]] top_pt_clustering
#        map[string, vector[float]] top_energy_clustering
#        map[string, vector[float]] top_children_dr
#
#        map[string, map[string, vector[float]]] fractional


cdef class ChildrenKinematics(SelectionTemplate):
    cdef childrenkinematics* tt

    cdef public dict res_kinematics
    cdef public dict spec_kinematics
    cdef public dict res_pdgid_kinematics
    cdef public dict spec_pdgid_kinematics

    cdef public dict res_decay_mode
    cdef public dict spec_decay_mode
    cdef public dict mass_clustering
    cdef public dict fractional

    cdef public dict dr_clustering
    cdef public dict top_pt_clustering
    cdef public dict top_energy_clustering
    cdef public dict top_children_dr

