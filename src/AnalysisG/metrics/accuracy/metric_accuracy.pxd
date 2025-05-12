# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp cimport int, float
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *
from AnalysisG.core.tools cimport *
from AnalysisG.core.roc cimport *

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+
   
    cdef struct cdata_t:
        int kfold
        vector[int] ntops_truth
        vector[vector[double]] ntop_score
        map[int, vector[double]] ntop_edge_accuracy
        map[int, map[int, vector[double]]] ntru_npred_matrix

    cdef cppclass collector(tools):
        collector() except+
        cdata_t* get_mode(string model, string mode, int epoch, int kfolds) except+ 

        void add_ntop_truth(string mode, string model, int epoch, int kfold, int data) except+
        void add_ntop_edge_accuracy(string mode, string model, int epoch, int kfold, int ntops, double data) except+
        void add_ntop_scores(string mode, string model, int epoch, int kfold, vector[double]* data) except+
        void add_ntru_ntop_scores(string mode, string model, int epoch, int kfold, int ntru, int ntop, double data) except+
        map[string, vector[cdata_t*]] get_plts() except+

        vector[string] model_names
        vector[string] modes
        vector[int] epochs
        vector[int] kfolds

cdef class AccuracyMetric(MetricTemplate):
    cdef accuracy_metric* mtr
    cdef collector* cl
    cdef public default_plt
    cdef public dict auc
