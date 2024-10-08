# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<models/RecursiveGraphNeuralNetwork.h>":
    cdef cppclass recursivegraphneuralnetwork(model_template):

        recursivegraphneuralnetwork(int rep, double drp) except+

        int _dx
        int _x
        int _output
        int _rep
        double res_mass
        double drop_out

        bool is_mc



cdef class RecursiveGraphNeuralNetwork(ModelTemplate):
    cdef recursivegraphneuralnetwork* rnn
