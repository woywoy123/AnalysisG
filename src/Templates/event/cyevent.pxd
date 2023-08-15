from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "event.h" namespace "CyTemplate":
    cdef cppclass CyEventTemplate:
        CyEventTemplate() except +

        string event_tree
        string event_tagging
        string implementation_name
        string commit_hash
        string pickle_string;
        signed int event_index
        float weight
        bool cached
        bool deprecated

        void Hash(string inpt) except+ nogil
        string Hash() except+ nogil
