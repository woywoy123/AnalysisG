# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport env, enc
from AnalysisG.core.notification cimport notification

cdef class Notification:

    def __cinit__(self): pass
    def __init__(self, str prx = ""): 
        self.ptr = new notification()
        self.ptr.prefix = enc(prx)

    def __dealloc__(self): del self.ptr
