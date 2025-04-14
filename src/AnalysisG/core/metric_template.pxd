# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "<templates/metric_template.h>" nogil:
    cdef cppclass metric_template:
        metric_template() except+ nogil
        string name
        map[string, string] run_names
        vector[string] variables

cdef class MetricTemplate:
    cdef metric_template* ptr
