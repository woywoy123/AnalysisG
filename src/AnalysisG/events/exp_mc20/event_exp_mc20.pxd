# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.events.exp_mc20.particle_exp_mc20 cimport *
from AnalysisG.core.event_template cimport *

cdef extern from "<exp_mc20/event.h>":

    cdef cppclass exp_mc20(event_template):
        exp_mc20() except+

        vector[particle_template*] Tops
        vector[particle_template*] TruthChildren
        vector[particle_template*] PhysicsTruth
        vector[particle_template*] Jets
        vector[particle_template*] Leptons
        vector[particle_template*] PhysicsDetector
        vector[particle_template*] Detector

        unsigned long long event_number
        float met_sum
        float met
        float phi
        float mu

cdef class ExpMC20(EventTemplate):
    cdef exp_mc20* exp
