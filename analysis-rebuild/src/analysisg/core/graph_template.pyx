# distutils: language = c++
# cython: language_level = 3

from libcpp cimport int
from analysisg.core.tools cimport *

cdef class GraphTemplate:
    def __cinit__(self):
        if type(self) is not GraphTemplate: return
        self.ptr = new graph_template()

    def __init__(self): pass
    def __dealloc__(self):
        if type(self) is not GraphTemplate: return
        del self.ptr

    def __hash__(self): return int(string(self.ptr.hash).substr(0, 8), 0)
    def __eq__(self, other):
        if not self.is_self(other): return False
        cdef GraphTemplate ev = other
        return self.ptr[0] == ev.ptr[0]

    def is_self(self, inpt) -> bool:
        if isinstance(inpt, GraphTemplate): return True
        return issubclass(inpt.__class__, GraphTemplate)

    @property
    def index(self): return self.ptr.index

    @property
    def Tree(self): return env(self.ptr.tree)
