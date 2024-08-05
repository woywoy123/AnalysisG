# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.particle_template cimport *

cdef extern from "<inference/gnn-particles.h>":

    cdef cppclass top_gnn(particle_template):
        top_gnn() except+

cdef class TopGNN(ParticleTemplate):
    cdef top_gnn* prt
