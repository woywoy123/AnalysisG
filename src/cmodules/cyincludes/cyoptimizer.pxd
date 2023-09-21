from cytypes cimport folds_t, data_t, metric_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../optimizer/optimizer.h" namespace "Optimizer":
    cdef cppclass CyEpoch nogil:
        CyEpoch() except +
        map[string, metric_t] metrics() except +
        map[int, map[string, data_t]] container


    cdef cppclass CyOptimizer nogil:
        CyOptimizer() except +
        void register_fold(const folds_t* inpt) except +
        void train_epoch_kfold(int epoch, int kfold, map[string, data_t]* data) except +
        void validation_epoch_kfold(int epoch, int kfold, map[string, data_t]* data) except +
        void evaluation_epoch_kfold(int epoch, int kfold, map[string, data_t]* data) except +

        void flush_train(vector[string], int) except +
        void flush_validation(vector[string], int) except +
        void flush_evaluation(vector[string]) except +

        vector[vector[string]] fetch_train(int kfold, int batch) except +
        vector[vector[string]] fetch_validation(int kfold, int batch) except +
        vector[vector[string]] fetch_evaluation(int batch) except +

        vector[string] check_train(vector[string], int) except +
        vector[string] check_validation(vector[string], int) except +
        vector[string] check_evaluation(vector[string]) except +

        map[string, int] fold_map() except +

        vector[int] use_folds
        map[int, CyEpoch*] epoch_train
        map[int, CyEpoch*] epoch_valid
        map[int, CyEpoch*] epoch_test





