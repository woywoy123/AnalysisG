# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "event.h":
    cdef cppclass event(selection_template):
        event() except +

cdef class Event(SelectionTemplate):
    cdef event* tt



