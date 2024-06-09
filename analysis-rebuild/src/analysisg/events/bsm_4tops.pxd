# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

from analysisg.core.particle_template cimport particle_template
from analysisg.core.event_template cimport event_template
from analysisg.core.event_template cimport EventTemplate


cdef extern from "bsm_4tops/particles.h":

    cdef cppclass top(particle_template):
        pass

    cdef cppclass children(particle_template):
        pass

    cdef cppclass truthjet(particle_template):
        pass

    cdef cppclass parton(particle_template):
        pass

    cdef cppclass jet(particle_template):
        pass

cdef extern from "bsm_4tops/event.h":

    cdef cppclass bsm_4tops(event_template):

        bsm_4tops() except+

        vector[string] leaves

        map[string, top*]      Tops
        map[string, children*] Children
        map[string, truthjet*] TruthJets
        map[string, jet*]      Jets
        map[string, parton*]   Partons


cdef class BSM4Tops(EventTemplate):

    pass
