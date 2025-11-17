# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.bsm_4tops.event_bsm_4tops cimport bsm_4tops
from AnalysisG.core.event_template cimport EventTemplate

cdef class BSM4Tops(EventTemplate):
    def __cinit__(self): 
        self.tt = new bsm_4tops()
        self.ptr = <event_template*>(self.tt)
    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    
    @property
    def reconstruct_nunu(self): return self.tt.reconstruct_nunu;
    @reconstruct_nunu.setter
    def reconstruct_nunu(self, bool val): self.tt.reconstruct_nunu = val
