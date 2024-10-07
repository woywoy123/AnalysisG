# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef class MET(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new met()
        self.tt = <met*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.missing_et = as_basic_dict(&self.tt.missing_et)

