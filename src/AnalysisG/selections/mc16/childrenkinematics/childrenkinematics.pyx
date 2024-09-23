# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef class ChildrenKinematics(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new childrenkinematics()
        self.tt = <childrenkinematics*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.res_kinematics         = as_dict(&self.tt.res_kinematics)
        self.spec_kinematics        = as_dict(&self.tt.spec_kinematics)
        self.mass_clustering        = as_dict(&self.tt.mass_clustering)
        self.dr_clustering          = as_dict(&self.tt.dr_clustering)
        self.top_pt_clustering      = as_dict(&self.tt.top_pt_clustering)
        self.top_energy_clustering  = as_dict(&self.tt.top_energy_clustering)
        self.top_children_dr        = as_dict(&self.tt.top_children_dr)

        self.res_pdgid_kinematics   = as_dict_dict(&self.tt.res_pdgid_kinematics)
        self.spec_pdgid_kinematics  = as_dict_dict(&self.tt.spec_pdgid_kinematics)
        self.res_decay_mode         = as_dict_dict(&self.tt.res_decay_mode)
        self.spec_decay_mode        = as_dict_dict(&self.tt.spec_decay_mode)
        self.fractional             = as_dict_dict(&self.tt.fractional)

