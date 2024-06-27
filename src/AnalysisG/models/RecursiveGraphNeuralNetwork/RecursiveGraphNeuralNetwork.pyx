# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.models.RecursiveGraphNeuralNetwork.RecursiveGraphNeuralNetwork cimport recursivegraphneuralnetwork

cdef class RecursiveGraphNeuralNetwork(ModelTemplate):
    def __cinit__(self): self.nn_ptr = new recursivegraphneuralnetwork()
    def __init__(self): pass
    def __dealloc__(self): del self.nn_ptr
