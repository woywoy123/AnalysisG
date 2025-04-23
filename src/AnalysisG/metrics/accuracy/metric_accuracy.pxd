# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp cimport int, float
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *
from AnalysisG.core.plotting cimport *

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+
  
cdef struct modelx_t:
    # - name - k - scores 
    map[string, map[int, vector[int   ]]] ntops_truth
    map[string, map[int, vector[double]]] edge_scores
    map[string, map[int, vector[vector[double]]]] ntop_score
    map[string, map[int, map[int, vector[double]]]] ntru_npred_ntop

cdef cppclass plt_roc_t:
    plt_roc_t() except+ nogil
    string model 
    string mode
    string variable

    int kfold 
    int epoch
    double auc

    vector[int] truth
    vector[vector[double]] scores



cdef class AccuracyMetric(MetricTemplate):
    cdef accuracy_metric* mtr

    # mode - epoch
    cdef map[string, map[int, modelx_t]] event_level
