# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.<particle-module>.<particle-name> cimport <particle-name>
from AnalysisG.core.particle_template cimport ParticleTemplate

cdef class <Python-Particle>(ParticleTemplate):

    def __cinit__(self): self.ptr = new <particle-name>()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    # do some stuff here

