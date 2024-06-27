# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.structs cimport particle_t

cdef extern from "<templates/particle_template.h>":

    cdef cppclass particle_template:
        particle_template() except +

        particle_template(particle_t* p) except +
        particle_template(double px, double py, double pz, double e) except +
        particle_template(double px, double py, double pz) except +

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

        double DeltaR(particle_template* p) except +

        bool is_lep
        bool is_nu
        bool is_b
        bool is_add
        bool lep_decay

        void to_cartesian() except +
        void to_polar() except +

        void add_leaf(string key, string leaf) except +

        bool operator == (particle_template& p) except +
        particle_template* operator+(particle_template* p) except +
        void iadd(particle_template* p) except +

        bool register_parent(particle_template* p) except +
        bool register_child(particle_template* p) except +

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
