# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from analysisg.cmodules.structs.structs cimport particle_t

cdef extern from "<particles/particles.h>":

    cdef cppclass particles:
        particles() except +

        particles(particle_t* p) except +
        particles(double px, double py, double pz, double e) except +
        particles(double px, double py, double pz) except +

        void e(double val) except +
        double e() except +

        double DeltaR(particles* p) except +

        void mass(double val) except +
        double mass() except +

        void pdgid(int val) except +
        int pdgid() except +

        void symbol(string val) except +
        string symbol() except +

        void charge(double val) except +
        double charge() except +

        bool is_lep() except +
        bool is_nu() except +
        bool is_b() except +
        bool is_add() except +
        bool lep_decay(vector[particle_t]*) except +

        double px() except +
        double py() except +
        double pz() except +

        void px(double val) except +
        void py(double val) except +
        void pz(double val) except +

        double pt() except +
        double eta() except +
        double phi() except +

        void pt(double val) except +
        void eta(double val) except +
        void phi(double val) except +

        void to_cartesian() except +
        void to_polar() except +

        string hash() except +
        void add_leaf(string key, string leaf) except +

        bool operator == (particles& p) except +
        particles* operator+(particles* p) except +
        void iadd(particles* p) except +

        bool register_parent(particles* p) except +
        bool register_child(particles* p) except +

        map[string, particles*] parents
        map[string, particles*] children
        map[string, string] leaves
        particle_t data

cdef class ParticleTemplate:
    cdef particles* ptr
    cdef list children
    cdef list parents

    cpdef ParticleTemplate clone(self)
    cpdef double DeltaR(self, ParticleTemplate inpt)
