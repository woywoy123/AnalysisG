# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.ssml_mc20.event_ssml_mc20 cimport event_ssml_mc20
from AnalysisG.core.event_template cimport EventTemplate

cdef class SSML_MC20(EventTemplate):

    def __cinit__(self): self.ptr = new event_ssml_mc20()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr


