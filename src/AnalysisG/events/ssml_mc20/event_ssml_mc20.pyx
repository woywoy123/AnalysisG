# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.ssml_mc20.event_ssml_mc20 cimport *
from AnalysisG.core.event_template cimport EventTemplate

cdef class SSML_MC20(EventTemplate):

    def __cinit__(self): 
        self.ev = new ssml_mc20()
        self.ptr = <event_template*>self.ev

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


