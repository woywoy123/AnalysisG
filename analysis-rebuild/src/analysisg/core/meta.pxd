# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from analysisg.core.structs cimport meta_t

cdef extern from "<meta/meta.h>":

    cdef cppclass meta:
        meta() except +

        meta_t meta_data

cdef class Meta:
    cdef meta* ptr
    cdef __meta__(self, meta* met)
