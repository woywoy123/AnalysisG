# distutils: language=c++
# cython: language_level = 3

from libcpp cimport int
from libcpp.vector cimport vector

from AnalysisG.core.event_template cimport *
from AnalysisG.core.tools cimport *

cdef class EventTemplate:
    def __cinit__(self):
        if type(self) is not EventTemplate: return
        self.ptr = new event_template()

    def __init__(self): pass
    def __dealloc__(self):
        if type(self) is not EventTemplate: return
        del self.ptr

    def __hash__(self): return int(string(self.ptr.hash).substr(0, 8), 0)
    def __eq__(self, other):
        if not self.is_self(other): return False
        cdef EventTemplate ev = other
        return self.ptr[0] == ev.ptr[0]

    def is_self(self, inpt) -> bool:
        if isinstance(inpt, EventTemplate): return True
        return issubclass(inpt.__class__, EventTemplate)

    @property
    def index(self) -> int: return self.ptr.index

    @index.setter
    def index(self, val: Union[str, double]):
        try: self.ptr.index = val
        except TypeError: self.ptr.leaves = [enc(val)]

    @property
    def weight(self) -> double: return self.ptr.weight

    @weight.setter
    def weight(self, val: Union[str, double]):
        try: self.ptr.weight = val
        except TypeError: self.ptr.leaves = [enc(val)]

    @property
    def Tree(self) -> str: return env(self.ptr.tree)

    @Tree.setter
    def Tree(self, str val): self.ptr.tree = enc(val)

    @property
    def Trees(self) -> list:
        cdef string x
        cdef vector[string] p = self.ptr.trees
        return [env(x) for x in p]

    @Trees.setter
    def Trees(self, val: Union[str, list]):
        cdef str i
        if isinstance(val, str): self.ptr.trees = [enc(val)]
        elif isinstance(val, list):
            for i in val: self.Trees = i
        else: pass

    @property
    def Branches(self) -> list:
        cdef string x
        cdef vector[string] p = self.ptr.branches
        return [env(x) for x in p]

    @Branches.setter
    def Branches(self, val: Union[str, list]):
        if isinstance(val, str): self.ptr.branches = [enc(val)]
        elif isinstance(val, list):
            for i in val: self.Branches = i
        else: pass

