# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.tools cimport *

cdef class <name>Metric(MetricTemplate):
    def __cinit__(self):
        self.ptr = new <name>_metric()
        self.mtr = <<name>_metric*>(self.ptr)

