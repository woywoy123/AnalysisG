# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.<event-module>.<event-name> cimport <event-name>
from AnalysisG.core.event_template cimport EventTemplate

cdef class <Python-Event>(EventTemplate):

    def __cinit__(self): self.ptr = new <event-name>()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    # do some stuff here

