# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.generators.optimizer cimport optimizer
from AnalysisG.core.model_template cimport ModelTemplate

cdef class Optimizer:
    def __cinit__(self): self.ev_ptr = new optimizer();
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr

