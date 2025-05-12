# distutils: language = c++
# cython: language_level = 3

from AnalysisG.core.plotting cimport *
from AnalysisG.core.tools cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool, float
from libcpp.map cimport map
from cython.operator cimport dereference as deref

cdef extern from "<plotting/roc.h>" nogil:
    cdef struct roc_t:
        int cls
        int kfold
        string model
        
        vector[double] _auc
        vector[vector[double]] tpr_
        vector[vector[double]] fpr_

        vector[vector[int]]*     truth
        vector[vector[double]]* scores

    cdef cppclass roc(plotting):
        roc() except+

        void build_ROC(string name, int kfold, vector[int]* label, vector[vector[double]]* scores) except+
        vector[roc_t*] get_ROC() except+

        map[string, map[int, vector[vector[double]]*]] roc_data
        map[string, map[int, vector[vector[int]]*]]    labels


cdef class ROC(TLine):
    cdef roc* rx

    cdef int num_cls
    cdef bool inits
    cdef bool verbose
    cdef public dict auc
    cdef public default_plt

    cdef void factory(self)
    cdef dict __compile__(self, bool raw = *)

