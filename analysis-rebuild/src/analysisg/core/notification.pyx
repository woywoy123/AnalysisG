# distutils: language = c++
# cython: language_level = 3

from analysisg.core.notification cimport notification

cdef class Notification:

    def __cinit__(self): pass
    def __init__(self): self.ptr = new notification()
    def __dealloc__(self): del self.ptr
