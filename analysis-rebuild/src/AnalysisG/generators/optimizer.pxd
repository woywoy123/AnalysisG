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

cdef class Optimizer:
    cdef optimizer* ev_ptr

