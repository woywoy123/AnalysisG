from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../Headers/Tools.h" namespace "Tools":
    pass

cdef extern from "../Headers/Particles.h" namespace "CyTemplate" nogil:
    cdef cppclass CyParticle nogil:
        CyParticle() nogil except +
        CyParticle(double px, double py, double pz, double e) nogil except + 
        CyParticle operator+(const CyParticle& p1) nogil except +   
        bool operator==(const CyParticle& p1) nogil except +   
        
        # Book keeping variables
        signed int index; 
        
        # State indicator
        bool _edited; 
        
        # Particle def 
        vector[signed int] _lepdef; 
        vector[signed int] _nudef;
        string Type; 

        # Getter Functions
        double px() nogil except +
        double py() nogil except +
        double pz() nogil except + 
        double pt() nogil except +
        double eta() except +
        double phi() except +
        double e() except +
        double Mass() except +
        double DeltaR(const CyParticle& p) except +
        signed int pdgid() except +
        double charge() except +
        string symbol() except +
        bool is_lep() except +
        bool is_nu() except +
        bool is_b() except +

        # Setter Functions
        void px(double val) except + 
        void py(double val) except + 
        void pz(double val) except +  
        void pt(double val) except + 
        void eta(double val) except + 
        void phi(double val) except +
        void e(double val) except + 
        void Mass(double val) except +
        void pdgid(signed int val) except +
        void charge(double val) except +

        string Hash() except +
        void _UpdateState() except +
