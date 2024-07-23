# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.models.<model-name>.<model-name> cimport <model-name>

cdef class <py-model-name>(ModelTemplate):
    def __cinit__(self): self.nn_ptr = new <model-name>()
    def __init__(self): pass
    def __dealloc__(self): del self.nn_ptr
