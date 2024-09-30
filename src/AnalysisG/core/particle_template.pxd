# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.structs cimport particle_t

cdef extern from "<templates/particle_template.h>" nogil:

    cdef cppclass particle_template:
        particle_template() except+ nogil

        particle_template(particle_t* p) except+ nogil
        particle_template(double px, double py, double pz, double e) except+ nogil
        particle_template(double px, double py, double pz) except+ nogil

        double mass
        double e

        double pt
        double eta
        double phi

        double px
        double py
        double pz

        int pdgid

        string hash
        string type
        string symbol
        double charge

        double DeltaR(particle_template* p) except+ nogil

        bool is_lep
        bool is_nu
        bool is_b
        bool is_add
        bool lep_decay

        void to_cartesian() except+ nogil
        void to_polar() except+ nogil

        void add_leaf(string key, string leaf) except+ nogil

        bool operator == (particle_template& p) except+ nogil
        particle_template* operator+(particle_template* p) except+ nogil
        void iadd(particle_template* p) except+ nogil

        bool register_parent(particle_template* p) except+ nogil
        bool register_child(particle_template* p) except+ nogil

        map[string, particle_template*] parents
        map[string, particle_template*] children
        map[string, string] leaves
        particle_t data

cdef class ParticleTemplate:
    cdef particle_template* ptr
    cdef list children
    cdef list parents

    cpdef ParticleTemplate clone(self)
    cpdef double DeltaR(self, ParticleTemplate inpt)
