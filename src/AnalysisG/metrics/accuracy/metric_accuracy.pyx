# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.tools cimport *

cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):
        self.ptr = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.ptr)

