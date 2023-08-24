from cytypes cimport meta_t, event_t, event_T
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../event/event.h" namespace "CyTemplate":
    cdef cppclass CyEventTemplate:
        CyEventTemplate() except +

        void ImportMetaData(meta_t meta) except +
        void Import(event_T event) except +
        event_T Export() except +

        string Hash() except +
        void addleaf(string key, string leaf) except +
        void addbranch(string key, string branch) except +
        void addtree(string key, string tree) except +

        void add_eventname(string event) except +

        bool operator == (CyEventTemplate* ev) except +

        event_t event
        meta_t meta

        map[string, string] leaves
        map[string, string] branches
        map[string, string] trees

