# distuils: language = c++
# cython: language_level = 3
from cyevent cimport CyEventTemplate
from libcpp.string cimport string
from libcpp.map cimport map

cdef class EventTemplate:
    cdef CyEventTemplate* ptr

    def __cinit__(self):
        self.ptr = new CyEventTemplate()

    def __init__(self):
        self.implementation_name = self.__class__.__name__

    def __dealloc__(self):
        del self.ptr

    @property
    def hash(self) -> str:
        return self.ptr.Hash().decode("UTF-8")

    @hash.setter
    def hash(self, str val):
        self.ptr.Hash(val.encode("UTF-8"))

