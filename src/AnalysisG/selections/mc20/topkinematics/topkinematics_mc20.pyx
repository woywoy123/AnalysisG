# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class TopKinematics(SelectionTemplate):
    def __cinit__(self):
        self.tt = new topkinematics()
        self.ptr = <selection_template*>self.tt

    def __init__(self): return

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        out = ()
        return TopKinematics, out

    cdef void transform_dict_keys(self):
        pass

