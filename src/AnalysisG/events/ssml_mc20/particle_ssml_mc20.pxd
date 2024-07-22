# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.particle_template cimport *

cdef extern from "<<particle-module>/<particle-name>.h>":

    cdef cppclass <particle-name>(particle_template):
        <particle-name>() except+
        float some_variable

cdef class <Python-Particle>(ParticleTemplate):
    pass
