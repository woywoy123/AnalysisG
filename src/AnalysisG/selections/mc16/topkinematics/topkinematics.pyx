# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *

cdef class TopKinematics(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topkinematics()
        self.tt = <topkinematics*>self.ptr
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    @property
    def res_top_kinematics(self): return self.tt.res_top_kinematics

    @property
    def spec_top_kinematics(self): return self.tt.spec_top_kinematics

    @property
    def mass_combi(self): return self.tt.mass_combi

    @property
    def deltaR(self): return self.tt.deltaR

