# distutils: language = c++
# cython: language_level = 3

from libcpp.map cimport map
from libcpp.string cimport string

from analysisg.core.io cimport IO
from analysisg.core.event_template cimport event_template

cdef extern from "<generators/eventgenerator.h>":
    cdef cppclass eventgenerator:

        eventgenerator() except +


cdef class EventGenerator(IO):

    cdef eventgenerator* ev_ptr
    cdef void event_compiler(self, event_template*)



