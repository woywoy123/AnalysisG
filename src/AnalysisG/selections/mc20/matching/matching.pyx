# distutils: language=c++
# cython: language_level=3

from libcpp.vector cimport vector
from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef class TopMatching(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new matching()
        self.tt = <matching*>self.ptr

    def __dealloc__(self): del self.tt

