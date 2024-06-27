# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string

cdef class SelectionTemplate:
    def __cinit__(self):
        if type(self) is not SelectionTemplate: return
        self.ptr = new selection_template()

    def __init__(self): pass

    def __dealloc__(self):
        if type(self) is not SelectionTemplate: return
        del self.ptr

    def __hash__(self):
        return int(string(self.ptr.hash).substr(0, 8), 0)


