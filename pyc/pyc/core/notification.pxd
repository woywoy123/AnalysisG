# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "<notification/notification.h>" nogil:

    cdef cppclass notification:

        void success(string) except+ nogil
        void warning(string) except+ nogil
        void failure(string) except+ nogil
        void info(string) except+ nogil

        string prefix
        int _warning
        int _failure
        int _success
        int _info
        bool bold
        bool shush

cdef class Notification:
    cdef notification* ptr
