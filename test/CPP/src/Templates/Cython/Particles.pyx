#distutils: language = c++
from Particles cimport CyParticle

cdef class ParticleTemplate:
    cdef CyParticle* _ptr

    def __cinit__(self):
        self._ptr = new CyParticle()

    def __dealloc__(self):
        del self._ptr

    @property 
    def index(self):
        return self._ptr.index
    
    @index.setter
    def index(self, int val):
        self._ptr.index = val

