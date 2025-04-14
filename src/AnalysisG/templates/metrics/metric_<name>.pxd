# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/<name>.h>":
    cdef cppclass <name>_metric(metric_template):
        <name>_metric() except+
    
cdef class <name>Metric(MetricTemplate):
    cdef <name>_metric* mtr
