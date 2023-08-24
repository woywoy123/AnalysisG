from libcpp.string cimport string
from cytypes cimport meta_t, event_t

cdef extern from "../abstractions/abstractions.h" namespace "CyTemplate":
    cdef cppclass CyEvent:
        CyEvent() except +
        void ImportMetaData(meta_t meta) except +
        void add_eventname(string event) except +
        string Hash() except +

        meta_t  meta
        event_t event
