# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.particle_template cimport *

cdef extern from "<inference/gnn-particles.h>":

    cdef cppclass top(particle_template):
        top() except+

    cdef cppclass zprime(particle_template):
        zprime() except+

cdef class Top(ParticleTemplate):
    cdef top* prt

cdef class ZPrime(ParticleTemplate):
    cdef zprime* prt
