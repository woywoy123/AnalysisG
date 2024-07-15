# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.particle_template cimport particle_template
from AnalysisG.events.bsm_4tops.particle_bsm_4tops cimport *
from AnalysisG.core.event_template cimport *

cdef extern from "<bsm_4tops/event.h>":

    cdef cppclass bsm_4tops(event_template):
        bsm_4tops() except+

        vector[particle_template*] Tops
        vector[particle_template*] Children
        vector[particle_template*] TruthJets
        vector[particle_template*] Jets
        vector[particle_template*] Electrons
        vector[particle_template*] Muons
        vector[particle_template*] DetectorObjects

        unsigned long long event_number
        float mu
        float met
        float phi


cdef class BSM4Tops(EventTemplate):
    pass
