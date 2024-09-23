# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict
from AnalysisG.core.selection_template cimport *

cdef class TopKinematics(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topkinematics()
        self.tt = <topkinematics*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.res_top_kinematics  = as_dict(&self.tt.res_top_kinematics)
        self.spec_top_kinematics = as_dict(&self.tt.spec_top_kinematics)
        self.mass_combi          = as_dict(&self.tt.mass_combi)
        self.deltaR              = as_dict(&self.tt.deltaR)
