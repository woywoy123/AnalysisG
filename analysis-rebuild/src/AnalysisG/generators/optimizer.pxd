# distutils: language=c++
# cython: language_level = 3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.model_template cimport model_template
from AnalysisG.core.graph_template cimport graph_template

cdef extern from "<generators/optimizer.h>":
    cdef cppclass optimizer:
        optimizer() except+

        void define_optimizer(string name) except +
        void define_model(model_template* model) except +
        void create_data_loader(vector[graph_template*]* data) except +
        void start() except +

cdef class Optimizer:
    cdef optimizer* ev_ptr

