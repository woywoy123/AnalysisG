# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "<notification/notification.h>":

    cdef cppclass notification:

        void success(string)
        void warning(string)
        void failure(string)
        void info(string)

        string prefix
        int _warning
        int _failure
        int _success
        int _info
        bool bold
        bool shush

cdef class Notification:
    cdef notification* ptr
