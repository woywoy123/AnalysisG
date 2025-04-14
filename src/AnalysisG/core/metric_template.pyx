# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string
from libcpp.map cimport map, pair
from libcpp.vector cimport vector

from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *

cdef class MetricTemplate:
    def __cinit__(self): self.ptr = new metric_template()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    def __name__(self): return env(self.ptr.name)

    @property
    def RunNames(self): 
        cdef map[string, string] o = self.ptr.run_names
        return as_basic_dict(&o)

    @RunNames.setter
    def RunNames(self, dict val): 
        cdef map[string, string] o
        as_map(val, &o)
        self.ptr.run_names = o

    @property
    def Variables(self): 
        cdef vector[string] o = self.ptr.variables
        return env_vec(&o)

    @Variables.setter
    def Variables(self, list val): 
        self.ptr.variables = enc_list(val)
