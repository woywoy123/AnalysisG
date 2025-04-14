# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+
    
cdef class AccuracyMetric(MetricTemplate):
    cdef accuracy_metric* mtr
