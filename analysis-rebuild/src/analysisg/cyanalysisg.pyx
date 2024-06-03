# cython: language_level = 3
from libcpp.string cimport string

from .cmodules cimport cmodules
from .cmodules.tools.tools cimport tools

cdef class cAnalysisG:

    def __cinit__(self): pass
    def __init__(self): pass

    cpdef void hello(self):
        cmodules.hashing(10)
        cdef string t = b"hello world"
        cdef tools* x = new tools()
        del x
