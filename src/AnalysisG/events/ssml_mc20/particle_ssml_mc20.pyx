# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.ssml_mc20.particle_ssml_mc20 cimport *
from AnalysisG.core.particle_template cimport ParticleTemplate

cdef class Jet(ParticleTemplate):

    def __cinit__(self):
        self.jt = new jet()
        self.ptr = <particle_template*>self.jt

    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class Lepton(ParticleTemplate):

    def __cinit__(self):
        self.lp = new lepton()
        self.ptr = <particle_template*>self.lp

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


