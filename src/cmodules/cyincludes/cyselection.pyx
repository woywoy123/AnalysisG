# distutils: language = c++
# cython: language_level = 3

from cyselection cimport CySelectionTemplate

cdef class SelectionTemplate:

    cdef CySelectionTemplate* ptr

    def __cinit__(self):
        self.ptr = new CySelectionTemplate()

    def __init__(self):
        pass

