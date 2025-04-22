# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.core.io cimport *

cdef extern from "<templates/metric_template.h>" nogil:
    cdef cppclass metric_template(tools, notification):
        metric_template() except+ nogil
        string name
        map[string, string] run_names
        vector[string] variables

cdef class MetricTemplate(Tools):
    cdef metric_template* mtx
    cdef public dict root_leaves
    cdef public dict root_fx
