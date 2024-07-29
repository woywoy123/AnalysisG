# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.exp_mc20.event_exp_mc20 cimport exp_mc20
from AnalysisG.core.event_template cimport EventTemplate

cdef class ExpMC20(EventTemplate):

    def __cinit__(self):
        self.exp = new exp_mc20()
        self.ptr = <event_template*>(self.exp)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr

