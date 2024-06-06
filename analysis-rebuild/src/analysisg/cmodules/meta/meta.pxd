# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from analysisg.cmodules.structs.structs cimport meta_t

cdef extern from "<meta/meta.h>":

    cdef cppclass meta:
        meta() except +



