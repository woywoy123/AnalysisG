# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/pagerank.h>":
    cdef cppclass pagerank_metric(metric_template):
        pagerank_metric() except+
    
cdef class PageRankMetric(MetricTemplate):
    cdef pagerank_metric* mtr
