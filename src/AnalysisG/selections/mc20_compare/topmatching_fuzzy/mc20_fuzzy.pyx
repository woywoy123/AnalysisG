# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef class Particle(ParticleTemplate):
    def __cinit__(self): self.ptr = new particle()
    def __dealloc__(self): del self.ptr

cdef class TopMatchingFuzzy(SelectionTemplate):
    def __cinit__(self):
        self.truth_tops = []
        self.top_children = []
        self.truth_jets = []
        self.jets_children = []
        self.jets_leptons = []
        self.ptr = new mc20_fuzzy()
        self.tt = <mc20_fuzzy*>self.ptr

    def __dealloc__(self): del self.tt

    cdef list make_particle(self, vector[particle*] px):
        cdef particle* p
        cdef Particle pi
        cdef list out = []
        for p in px:
            pi = Particle()
            pi.set_particle(p)
            out.append(pi)
        return out

    cdef void transform_dict_keys(self):
        cdef packet_t v
        for v in self.tt.output:
            self.truth_tops.append(self.make_particle(v.truth_tops))
            self.top_children.append(self.make_particle(v.children_tops))
            self.truth_jets.append(self.make_particle(v.truth_jets))
            self.jets_children.append(self.make_particle(v.jets_children))
            self.jets_leptons.append(self.make_particle(v.jets_leptons))
