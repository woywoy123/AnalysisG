# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.models.RecursiveGraphNeuralNetwork.RecursiveGraphNeuralNetwork cimport recursivegraphneuralnetwork

cdef class RecursiveGraphNeuralNetwork(ModelTemplate):
    def __cinit__(self):
        self.rnn = new recursivegraphneuralnetwork(26, 0.1)
        self.nn_ptr = self.rnn

    def __init__(self): pass
    def __dealloc__(self): del self.nn_ptr

    @property
    def dx(self): return self.rnn._dx
    @dx.setter
    def dx(self, int val): self.rnn._dx = val

    @property
    def x(self): return self.rnn._x
    @x.setter
    def x(self, int val): self.rnn._x = val

    @property
    def output(self): return self.rnn._output
    @output.setter
    def output(self, int val): self.rnn._output = val

    @property
    def rep(self): return self.rnn._rep
    @rep.setter
    def rep(self, int val): self.rnn._rep = val

    @property
    def NuR(self): return self.rnn.NuR
    @NuR.setter
    def NuR(self, bool val): self.rnn.NuR = val

    @property
    def is_mc(self): return self.rnn.is_mc
    @is_mc.setter
    def is_mc(self, bool val): self.rnn.is_mc = val

    @property
    def res_mass(self): return self.rnn.res_mass
    @res_mass.setter
    def res_mass(self, double val): self.rnn.res_mass = val

    @property
    def drop_out(self): return self.rnn.drop_out
    @drop_out.setter
    def drop_out(self, double val): self.rnn.drop_out = val

