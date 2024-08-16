# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.models.Experimental.Experimental cimport experimental

cdef class Experimental(ModelTemplate):
    def __cinit__(self):
        self.rnn = new experimental()
        self.nn_ptr = self.rnn

    def __init__(self): pass
    def __dealloc__(self): del self.nn_ptr

    @property
    def dx(self): return self.rnn._dxin
    @dx.setter
    def dx(self, int val): self.rnn._dxin = val

    @property
    def x(self): return self.rnn._xin
    @x.setter
    def x(self, int val): self.rnn._xin = val

    @property
    def is_mc(self): return self.rnn.is_mc
    @is_mc.setter
    def is_mc(self, bool val): self.rnn.is_mc = val

    @property
    def drop_out(self): return self.rnn.drop_out
    @drop_out.setter
    def drop_out(self, double val): self.rnn.drop_out = val

