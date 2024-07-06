# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict
from AnalysisG.core.selection_template cimport *

cdef class TopKinematics(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topkinematics()
        self.tt = <topkinematics*>self.ptr

    def __init__(self, v1 = None, v2 = None, v3 = None, v4 = None):
        if v1 is None: return
        self.res_top_kinematics  = v1
        self.spec_top_kinematics = v2
        self.mass_combi          = v3
        self.deltaR              = v4

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        out = (
            self.res_top_kinematics, self.spec_top_kinematics,
            self.mass_combi, self.deltaR,
        )
        return TopKinematics, out


    cdef void transform_dict_keys(self):
        self.res_top_kinematics  = as_dict(&self.tt.res_top_kinematics)
        self.spec_top_kinematics = as_dict(&self.tt.spec_top_kinematics)
        self.mass_combi          = as_dict(&self.tt.mass_combi)
        self.deltaR              = as_dict(&self.tt.deltaR)
