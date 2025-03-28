# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.tools cimport *

cdef class Neutrino(ParticleTemplate):
    def __cinit__(self):
        self._bquark = None
        self._lepton = None
        self.ptr = new neutrino()
    def __dealloc__(self): del self.ptr

    @property
    def distance(self): return (<neutrino*>(self.ptr)).min

    @distance.setter
    def distance(self, val): (<neutrino*>(self.ptr)).min = val

    @property
    def bquark(self):
        if self._bquark is not None: return self._bquark
        cdef Particle prt = Particle()
        cdef particle_template* bq = (<neutrino*>(self.ptr)).bquark
        if bq is NULL: return None
        prt.set_particle(bq)
        (<neutrino*>(self.ptr)).bquark = new particle_template((<neutrino*>(self.ptr)).bquark, False)
        self._bquark = prt
        return prt

    @property
    def lepton(self):
        if self._lepton is not None: return self._lepton
        cdef Particle prt = Particle()
        cdef particle_template* lp = (<neutrino*>(self.ptr)).lepton
        if lp is NULL: return None
        prt.set_particle(lp)
        (<neutrino*>(self.ptr)).lepton = new particle_template((<neutrino*>(self.ptr)).lepton, False)
        self._lepton = prt
        return prt

    @bquark.setter
    def bquark(self, p):
        if p is None: return
        self._bquark = p

    @lepton.setter
    def lepton(self, p):
        if p is None: return
        self._lepton = p

cdef class Particle(ParticleTemplate):
    def __cinit__(self): self.ptr = new particle()
    def __dealloc__(self): del self.ptr

cdef class Event:
    cdef list build_nu(self, vector[neutrino*] inpt):
        cdef neutrino* i
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
        cdef int x
        cdef Event ev
        cdef pair[string, event_data] itr
        cdef combinatorial* tt = <combinatorial*>(self.ptr)

        for itr in tt.output:
            ev = Event()
            ev.delta_met        = itr.second.delta_met
            ev.delta_metnu      = itr.second.delta_metnu
            ev.observed_met     = itr.second.observed_met
            ev.neutrino_met     = itr.second.neutrino_met
            ev.cobs_neutrinos = []
            ev.cmet_neutrinos = []
            ev.robs_neutrinos = []
            ev.rmet_neutrinos = []


            ev.truth_neutrinos  = ev.build_nu(itr.second.truth_neutrinos)
            for x in range(itr.second.cobs_neutrinos.size()):
                ev.cobs_neutrinos.append(ev.build_nu(itr.second.cobs_neutrinos[x]))

            for x in range(itr.second.cmet_neutrinos.size()):
                ev.cmet_neutrinos.append(ev.build_nu(itr.second.cmet_neutrinos[x]))


            for x in range(itr.second.robs_neutrinos.size()):
                ev.robs_neutrinos.append(ev.build_nu(itr.second.robs_neutrinos[x]))

            for x in range(itr.second.rmet_neutrinos.size()):
                ev.rmet_neutrinos.append(ev.build_nu(itr.second.rmet_neutrinos[x]))

            ev.bquark           = ev.build_particle(itr.second.bquark)
            ev.lepton           = ev.build_particle(itr.second.lepton)
            ev.tops             = ev.build_particle(itr.second.tops)
            self.events[env(itr.first)] = ev
