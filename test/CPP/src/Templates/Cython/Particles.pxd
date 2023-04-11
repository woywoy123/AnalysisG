cdef extern from "../Headers/Particles.h" namespace "CyTemplate":
    cdef cppclass CyParticle:
        CyParticle() except +

        # Book keeping variables
        signed int index; 


