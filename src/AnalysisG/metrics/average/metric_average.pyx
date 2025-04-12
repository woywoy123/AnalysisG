# distutils: language=c++
# cython: language_level=3

cdef class AverageMetric(MetricTemplate):
    def __cinit__(self):
        self.ptr = new average_metric()
        self.mtr = <average_metric*>(self.ptr)
