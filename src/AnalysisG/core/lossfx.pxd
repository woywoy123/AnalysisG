# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport cython.operator

cdef extern from "<templates/fx_enums.h>":

    cdef cppclass optimizer_params_t:

        void operator()()
        string optimizer

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
