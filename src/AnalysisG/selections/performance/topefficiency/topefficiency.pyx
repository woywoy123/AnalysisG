# distutils: language=c++
# cython: language_level=3

from AnalysisG.selections.performance.topefficiency.topefficiency cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *

cdef class TopEfficiency(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topefficiency()
        self.tt = <topefficiency*>self.ptr

    def __dealloc__(self): del self.tt


