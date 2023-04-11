cdef extern from "../CXX/Event.cxx":
    pass

cdef extern from "openssl/sha.h":
    pass

cdef extern from "../../Tools/CXX/Tools.cxx":
    pass

from libcpp.string cimport string
from libcpp cimport bool 

cdef extern from "../Headers/Event.h" namespace "Sample":
    cdef cppclass Event:
        # Constructor 
        Event() except +
        
        # Functions 
        void MakeHash(); 
        
        # Class Attributes
        string Hash; 
        signed int EventIndex;
        bool Compiled; 
        bool Train; 
