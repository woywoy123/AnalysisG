from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "particle.h" namespace "CyTemplate":
    cdef cppclass CyParticleTemplate:
        CyParticleTemplate() except +
        CyParticleTemplate(double px, double py, double pz, double e) except +
        CyParticleTemplate(double px, double py, double pz) except +

        double e() except +
        void e(double val) except +

        double mass() except +
        void mass(double val) except +

        int pdgid() except +
        void pdgid(int val) except +

        string symbol() except +
        void symbol(string val) except +

        double charge() except +
        void charge(double val) except +

        bool is_lep() except +
        bool is_nu() except +
        bool is_b() except +
        bool is_add() except +
        bool lep_decay() except +

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

        string hash() except +
        void addleaf(string key, string leaf) except +

        CyParticleTemplate* operator+(CyParticleTemplate* p) except +
        bool operator==(CyParticleTemplate* p) except +
        void iadd(CyParticleTemplate* p) except +

        string type
        int index
        vector[int] lepdef
        vector[int] nudef
        map[string, CyParticleTemplate*] children
        map[string, CyParticleTemplate*] parent
        map[string, string] leaves

