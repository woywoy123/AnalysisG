from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

cdef extern from "../plotting/plotting.h":
    struct roc_t:
        map[int, float] auc
        map[int, vector[float]] tpr
        map[int, vector[float]] fpr
        map[int, vector[float]] threshold
        vector[vector[float]] truth
        vector[vector[float]] pred
        vector[vector[int]] confusion



