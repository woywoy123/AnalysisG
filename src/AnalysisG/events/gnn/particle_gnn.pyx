# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.particle_template cimport ParticleTemplate
from AnalysisG.events.gnn.particle_gnn cimport *

cdef class Top(ParticleTemplate):

    def __cinit__(self):
        self.prt = new top()
        self.ptr = <particle_template*>(self.prt)

    def __init__(self): pass
    def __dealloc__(self): del self.prt

cdef class ZPrime(ParticleTemplate):

    def __cinit__(self):
        self.prt = new zprime()
        self.ptr = <particle_template*>(self.prt)

    def __init__(self): pass
    def __dealloc__(self): del self.prt


