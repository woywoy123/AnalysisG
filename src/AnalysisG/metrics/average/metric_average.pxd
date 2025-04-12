# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/average.h>":
    cdef cppclass average_metric(metric_template):
        average_metric() except+
    
cdef class AverageMetric(MetricTemplate):
    cdef average_metric* mtr
