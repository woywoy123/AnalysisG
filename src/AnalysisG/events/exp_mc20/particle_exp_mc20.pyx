# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.exp_mc20.particle_exp_mc20 cimport *
from AnalysisG.core.particle_template cimport ParticleTemplate

cdef class Top(ParticleTemplate):

    def __cinit__(self):
        self.p = new top()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class Child(ParticleTemplate):
    def __cinit__(self):
        self.p = new child()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


cdef class PhysicsDetector(ParticleTemplate):
    def __cinit__(self):
        self.p = new physics_detector()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


cdef class PhysicsTruth(ParticleTemplate):
    def __cinit__(self):
        self.p = new physics_truth()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class Electron(ParticleTemplate):
    def __cinit__(self):
        self.p = new electron()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


cdef class Muon(ParticleTemplate):
    def __cinit__(self):
        self.p = new muon()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr


cdef class Jet(ParticleTemplate):
    def __cinit__(self):
        self.p = new jet()
        self.ptr = <particle_template*>(self.p)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr

