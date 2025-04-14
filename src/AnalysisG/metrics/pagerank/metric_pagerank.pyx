# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.tools cimport *

cdef class PageRankMetric(MetricTemplate):
    def __cinit__(self):
        self.ptr = new pagerank_metric()
        self.mtr = <pagerank_metric*>(self.ptr)

