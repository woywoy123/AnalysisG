# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<RecursiveGraphNeuralNetwork.h>":
    cdef cppclass recursivegraphneuralnetwork(model_template):

        recursivegraphneuralnetwork() except+

        int _dx
        int _rep

        bool GeV
        bool NuR


cdef class RecursiveGraphNeuralNetwork(ModelTemplate):
    pass
