# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.particle_template cimport *

cdef extern from "<ssml_mc20/particles.h>":

    cdef cppclass jet(particle_template):
        jet() except+

        bool gn2_btag_65
        bool gn2_btag_70
        bool gn2_btag_77
        bool gn2_btag_85
        bool gn2_btag_90

    cdef cppclass lepton(particle_template):
        lepton() except+

cdef class Jet(ParticleTemplate):
    cdef jet* jt

cdef class Lepton(ParticleTemplate):
    cdef lepton* lp




