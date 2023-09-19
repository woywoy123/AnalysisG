from cytypes cimport folds_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../optimizer/optimizer.h" namespace "Optimizer":
    cdef cppclass CyOptimizer nogil:
        CyOptimizer() except +
        void register_fold(const folds_t* inpt) except +

        vector[vector[string]] fetch_train(int kfold, int batch) except +
        vector[string] check_train(vector[string], int) except +
        void flush_train(vector[string], int) except +

        vector[vector[string]] fetch_validation(int kfold, int batch) except +
        vector[string] check_validation(vector[string], int) except +
        void flush_validation(vector[string], int) except +

        map[string, int] fold_map() except +

        vector[int] use_folds
