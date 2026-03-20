# distutils: language=c++
# cython: language_level=3
from cython.operator cimport dereference as dref
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *
from tqdm import tqdm

cdef class MultiSol(SelectionTemplate):
    def __dealloc__(self): del self.tt
    def __cinit__(self):
        self.ptr = new multisol()
        self.tt = <multisol*>self.ptr
    def Postprocessing(self): pass


