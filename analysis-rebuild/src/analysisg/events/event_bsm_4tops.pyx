# distutils: language = c++
# cython: language_level = 3

from analysisg.events.event_bsm_4tops cimport bsm_4tops
from analysisg.core.event_template cimport EventTemplate

cdef class BSM4Tops(EventTemplate):

    def __cinit__(self): self.ptr = new bsm_4tops()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

