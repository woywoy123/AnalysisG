# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+
   

cdef struct modelx_t:
    map[int, vector[vector[double]]] ntop_score

cdef struct truth_t:
    vector[int] kfold
    vector[int] ntops

cdef class AccuracyMetric(MetricTemplate):
    cdef accuracy_metric* mtr

    # var - mode - data
    cdef map[string, map[string, truth_t]] truth_data

    # var - mode - model - 
    cdef map[string, map[string, map[string, modelx_t]]] event_acc


