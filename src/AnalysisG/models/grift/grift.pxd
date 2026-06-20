# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<models/grift.h>":
    cdef cppclass grift(model_template):

        grift() except+

        int _xrec
        int _xin
        double drop_out
        bool is_mc

cdef class Grift(ModelTemplate):
    cdef grift* rnn
