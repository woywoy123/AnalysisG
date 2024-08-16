# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<models/Experimental.h>":
    cdef cppclass experimental(model_template):

        experimental() except+

        int _dxin
        int _xin
        double drop_out
        bool is_mc

cdef class Experimental(ModelTemplate):
    cdef experimental* rnn
