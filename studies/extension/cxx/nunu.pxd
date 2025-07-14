# distutils: language = c++

cdef extern from "particle.h":
    cdef cppclass particle:
        particle(double px, double py, double pz, double e); 
        double p() 
        double p2() 
        double m()
        double m2()
        double beta()
        double beta2()
        double phi()
        double theta()

        double px 
        double py
        double pz
        double e  
        double d

cdef extern from "nunu.h":
    cdef cppclass nunu:
        nunu(
             double b1_px, double b1_py, double b1_pz, double b1_e,
             double l1_px, double l1_py, double l1_pz, double l1_e, 
             double b2_px, double b2_py, double b2_pz, double b2_e,
             double l2_px, double l2_py, double l2_pz, double l2_e, 
             double mt1  , double mt2  , double mw1  , double mw2
             ) except +

        void _clear(); 
        void get_misc()
        void get_nu(particle** nu1, particle** nu2, int lx); 
        int generate(double metx, double mety, double metz)


cdef class Particle:
    cdef particle* ptr


cdef class NuNu:
    cdef nunu* nu
