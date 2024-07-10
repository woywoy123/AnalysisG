# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *

cdef class <selection-name-python>(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new <selection-name>()
        self.tt = <<selection-name>*>self.ptr

    def __init__(self): return

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        out = ()
        return <selection-name-python>, out

    cdef void transform_dict_keys(self):
        #convert map keys to python string
        pass

