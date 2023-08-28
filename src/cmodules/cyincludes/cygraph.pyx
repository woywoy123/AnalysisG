# distutils: language = c++
# cython: language_level = 3

from cygraph cimport CyGraphTemplate

cdef class GraphTemplate:

    cdef CyGraphTemplate* ptr

    def __cinit__(self):
        self.ptr = new CyGraphTemplate()

    def __init__(self):
        pass

