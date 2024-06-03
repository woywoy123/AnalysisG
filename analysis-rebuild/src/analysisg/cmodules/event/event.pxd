# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "<event/event.h>":

    cdef cppclass event:
        event() except +

