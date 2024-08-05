# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.particle_template cimport ParticleTemplate
from AnalysisG.events.gnn.particle_gnn cimport *

cdef class TopGNN(ParticleTemplate):

    def __cinit__(self):
        self.prt = new top_gnn()
        self.ptr = <particle_template*>(self.prt)

    def __init__(self): pass
    def __dealloc__(self): del self.prt

