# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.models.Grift.Grift cimport grift

cdef class Grift(ModelTemplate):
    def __cinit__(self):
        self.rnn = new grift()
        self.nn_ptr = self.rnn

    def __init__(self): pass
    def __dealloc__(self): del self.rnn

    @property
    def xrec(self): return self.rnn._xrec
    @xrec.setter
    def xrec(self, int val): self.rnn._xrec = val

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

    @property
    def PageRank(self): return self.rnn.pagerank
    @PageRank.setter
    def PageRank(self, bool v): self.rnn.pagerank = v


