from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "event.h" namespace "CyTemplate":
    cdef cppclass CyEventTemplate:
        CyEventTemplate()

        int event_index
        double weight

        bool cached
        bool deprecated

        string event_tree
        string event_tagging
        string event_name
        string commit_hash
        string pickle_string;

        string Hash() except +
        void Hash(string inpt) except +
        void addleaf(string key, string leaf) except +
        void addbranch(string key, string branch) except +
        void addtree(string key, string tree) except +

        bool operator == (CyEventTemplate* ev) except +

        map[string, string] leaves
        map[string, string] branches
        map[string, string] trees


