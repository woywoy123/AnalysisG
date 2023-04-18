from libcpp.string cimport string 
from libcpp cimport bool

cdef extern from "../Headers/Tools.h" namespace "Tools":
    pass

cdef extern from "../Headers/Event.h" namespace "CyTemplate" nogil:
    cdef cppclass CyEvent nogil:
        CyEvent() nogil except +
        
        signed int index; 
        double weight; 
        bool deprecated; 
        string tree; 
        string commit_hash; 
        
        void Hash(string val) nogil except +
        string Hash() nogil except +
