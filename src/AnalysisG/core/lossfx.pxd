# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool, int
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport cython.operator

cdef extern from "<structs/optimizer.h>" nogil:

    cdef cppclass optimizer_params_t:

        optimizer_params_t() except+ nogil
        string optimizer
        string scheduler

        double gamma
        unsigned int step_size

        double lr
        double lr_decay
        double weight_decay
        double initial_accumulator_value
        double eps
        double tolerance_grad
        double tolerance_change
        double alpha
        double momentum
        double dampening

        bool amsgrad
        bool centered
        bool nesterov

        int max_iter
        int max_eval
        int history_size

        vector[float] beta_hack

cdef class OptimizerConfig:

    cdef optimizer_params_t* params
