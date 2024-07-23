# distutils: language=c++
# cython: language_level=3

from libcpp cimport int, bool
from AnalysisG.core.model_template cimport model_template, ModelTemplate

cdef extern from "<models/<model-name>.h>":
    cdef cppclass <model-name>(model_template):

        <model-name>() except+

        int _dx
        int _rep

        bool GeV
        bool NuR


cdef class <py-model-name>(ModelTemplate):
    pass
