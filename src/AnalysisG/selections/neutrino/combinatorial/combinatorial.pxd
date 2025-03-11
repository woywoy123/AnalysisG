# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.selection_template cimport *

cdef extern from "combinatorial.h":
    cdef cppclass neutrino(particle_template):
        neutrino() except+
        particle_template* bquark
        particle_template* lepton
        double min

    cdef cppclass particle(particle_template):
        particle() except+

    struct event_data:
        double delta_met
        double delta_metnu
        double observed_met
        double neutrino_met

        vector[neutrino*] truth_neutrinos

        vector[vector[neutrino*]] cobs_neutrinos
        vector[vector[neutrino*]] cmet_neutrinos

        vector[vector[neutrino*]] robs_neutrinos
        vector[vector[neutrino*]] rmet_neutrinos

        vector[particle*] bquark
        vector[particle*] lepton
        vector[particle*] tops

    cdef cppclass combinatorial(selection_template):
        combinatorial() except +
        map[string, event_data] output

cdef class Particle(ParticleTemplate):
    pass

cdef class Neutrino(ParticleTemplate):
    cdef Particle _bquark
    cdef Particle _lepton

cdef class Event:
    cdef list build_nu(self, vector[neutrino*] inpt)
    cdef list build_particle(self, vector[particle*] inpt)
    cdef public double delta_met
    cdef public double delta_metnu
    cdef public double observed_met
    cdef public double neutrino_met
    cdef public list truth_neutrinos

    cdef public list cobs_neutrinos
    cdef public list cmet_neutrinos

    cdef public list robs_neutrinos
    cdef public list rmet_neutrinos

    cdef public list bquark
    cdef public list lepton
    cdef public list tops

cdef class Combinatorial(SelectionTemplate):
    cdef public dict events
