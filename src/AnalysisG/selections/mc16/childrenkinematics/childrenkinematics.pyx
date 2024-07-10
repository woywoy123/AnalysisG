# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef class ChildrenKinematics(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new childrenkinematics()
        self.tt = <childrenkinematics*>self.ptr

    def __init__(self, data = None):
        if data is None: return
        self.res_kinematics = data["res_kinematics"]
        self.spec_kinematics = data["spec_kinematics"]
        self.mass_clustering = data["mass_clustering"]
        self.dr_clustering = data["dr_clustering"]
        self.top_pt_clustering = data["top_pt_clustering"]
        self.top_energy_clustering = data["top_energy_clustering"]
        self.top_children_dr = data["top_children_dr"]
        self.res_pdgid_kinematics = data["res_pdgid_kinematics"]
        self.spec_pdgid_kinematics = data["spec_pdgid_kinematics"]
        self.res_decay_mode = data["res_decay_mode"]
        self.spec_decay_mode = data["spec_decay_mode"]
        self.fractional = data["fractional"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        out = ({
            "res_kinematics":self.res_kinematics,
            "spec_kinematics":self.spec_kinematics,
            "mass_clustering":self.mass_clustering,
            "dr_clustering":self.dr_clustering,
            "top_pt_clustering":self.top_pt_clustering,
            "top_energy_clustering":self.top_energy_clustering,
            "top_children_dr":self.top_children_dr,
            "res_pdgid_kinematics":self.res_pdgid_kinematics,
            "spec_pdgid_kinematics":self.spec_pdgid_kinematics,
            "res_decay_mode":self.res_decay_mode,
            "spec_decay_mode":self.spec_decay_mode,
            "fractional":self.fractional
        }, )
        return ChildrenKinematics, out

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

