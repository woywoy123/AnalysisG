# distutils: language=c++
# cython: language_level=3

from libcpp.vector cimport vector
from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef class Particle(ParticleTemplate):
    def __cinit__(self): self.ptr = new particle()
    def __dealloc__(self):
        if not self.is_owner: return
        del self.ptr

    @property
    def top_hash(self):
        cdef particle* ptx = <particle*>(self.ptr)
        return env(ptx.root_hash)

    @top_hash.setter
    def top_hash(self, val):
        cdef particle* ptx = <particle*>(self.ptr)
        ptx.root_hash = enc(val)

cdef class TopMatching(SelectionTemplate):
    def __cinit__(self):
        self.truth_tops = []
        self.top_children = []
        self.truth_jets = []
        self.jets_children = []
        self.jets_leptons = []

        self.ptr = new matching()
        self.tt = <matching*>self.ptr

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

