# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.tools cimport *


cdef class Neutrino(ParticleTemplate):
    def __cinit__(self): self.ptr = new nu()
    def __dealloc__(self): del self.ptr

    @property
    def distance(self):
        cdef nu* lx = <nu*>(self.ptr)
        return lx.min

    @property
    def expected_masses(self):
        cdef nu* lx = <nu*>(self.ptr)
        return [lx.exp_tmass, lx.exp_wmass]

    @property
    def idx(self):
        cdef nu* lx = <nu*>(self.ptr)
        return lx.idx

cdef class Particle(ParticleTemplate):
    def __cinit__(self): self.ptr = new particle()
    def __dealloc__(self): del self.ptr

cdef class Event:
    cdef list build_nu(self, vector[nu*] inpt):
        cdef nu* i
        cdef Neutrino nux
        cdef list out = []
        for i in inpt:
            nux = Neutrino()
            nux.set_particle(i)
            out.append(nux)
        return out

    cdef list build_particle(self, vector[particle*] inpt):
        cdef particle* i
        cdef Particle nux
        cdef list out = []
        for i in inpt:
            nux = Particle()
            nux.set_particle(i)
            out.append(nux)
        return out

cdef class Combinatorial(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new combinatorial()
        self.events = {}

    def __dealloc__(self): del self.ptr

    cdef void transform_dict_keys(self):
        cdef Event ev
        cdef pair[string, event_data] itr
        cdef combinatorial* tt = <combinatorial*>(self.ptr)
        for itr in tt.output:
            ev = Event()
            ev.delta_met        = itr.second.delta_met
            ev.delta_metnu      = itr.second.delta_metnu
            ev.observed_met     = itr.second.observed_met
            ev.neutrino_met     = itr.second.neutrino_met

            ev.truth_neutrinos  = ev.build_nu(itr.second.truth_neutrinos)
            ev.cobs_neutrinos   = ev.build_nu(itr.second.cobs_neutrinos)
            ev.cmet_neutrinos   = ev.build_nu(itr.second.cmet_neutrinos)

            ev.robs_neutrinos   = ev.build_nu(itr.second.robs_neutrinos)
            ev.rmet_neutrinos   = ev.build_nu(itr.second.rmet_neutrinos)

            ev.bquark           = ev.build_particle(itr.second.bquark)
            ev.lepton           = ev.build_particle(itr.second.lepton)
            ev.tops             = ev.build_particle(itr.second.tops)
            self.events[env(itr.first)] = ev
