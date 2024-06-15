# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

from AnalysisG.core.particle_template cimport particle_template
from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.event_template cimport EventTemplate


cdef extern from "bsm_4tops/particles.h":

    cdef cppclass top(particle_template):
        top() except+

    cdef cppclass top_children(particle_template):
        top_children() except+

    cdef cppclass truthjet(particle_template):
        truthjet() except+

    cdef cppclass truthjetparton(particle_template):
        truthjetparton() except+

    cdef cppclass jet(particle_template):
        jet() except+

    cdef cppclass jetparton(particle_template):
        jetparton() except+

    cdef cppclass electron(particle_template):
        electron() except+

    cdef cppclass muon(particle_template):
        muon() except+

cdef extern from "bsm_4tops/event.h":

    cdef cppclass bsm_4tops(event_template):
        bsm_4tops() except+
        vector[string] leaves

cdef class BSM4Tops(EventTemplate):
    pass
