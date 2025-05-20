# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "<notification/notification.h>" nogil:

    cdef cppclass notification:

        void success(string) except+ nogil
        void warning(string) except+ nogil
        void failure(string) except+ nogil
        void info(string)    except+ nogil
        void progressbar(float prg, string title) except+ nogil
        void progressbar1(vector[unsigned long long]* threads, unsigned long long* l, string* title) except+ nogil
        void progressbar2(vector[unsigned long long]* threads, unsigned long long* l, string* title) except+ nogil
        void progressbar3(vector[unsigned long long]* threads, vector[unsigned long long]* l, vector[string*]* title) except+ nogil

        string prefix
        int _warning
        int _failure
        int _success
        int _info
        bool bold
        bool shush

cdef class Notification:
    cdef notification* ptr
