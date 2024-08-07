# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<models/Experimental.h>":
    cdef cppclass experimental(model_template):

        experimental(int rep, double drp) except+

        int _dx
        int _x
        int _output
        int _rep
        double res_mass
        double drop_out

        bool is_mc

cdef class Experimental(ModelTemplate):
    cdef experimental* rnn
