# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.selection_template cimport *

cdef extern from "combinatorial.h":
    cdef cppclass nu(particle_template):
        nu() except+
        double exp_tmass
        double exp_wmass
        double min
        long idx

    cdef cppclass particle(particle_template):
        particle() except+

    struct event_data:
        double delta_met
        double delta_metnu
        double observed_met
        double neutrino_met

        vector[nu*] truth_neutrinos

        vector[nu*] cobs_neutrinos
        vector[nu*] cmet_neutrinos

        vector[nu*] robs_neutrinos
        vector[nu*] rmet_neutrinos

        vector[particle*] bquark
        vector[particle*] lepton
        vector[particle*] tops

    cdef cppclass combinatorial(selection_template):
        combinatorial() except +
        map[string, event_data] output

cdef class Neutrino(ParticleTemplate):
    cdef void set_particle(self, nu* ox)

cdef class Particle(ParticleTemplate):
    cdef void set_particle(self, particle* ox)

cdef class Event:
    cdef list build_nu(self, vector[nu*] inpt)
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
