from cytypes cimport meta_t, event_t
from cycode cimport CyCode

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../event/event.h" namespace "CyTemplate":
    cdef cppclass CyEventTemplate:
        CyEventTemplate() except +

        void ImportMetaData(meta_t meta) except +
        void Import(event_t event) except +
        event_t Export() except +

        string Hash() except +
        void addleaf(string key, string leaf) except +
        void addbranch(string key, string branch) except +
        void addtree(string key, string tree) except +

        void set_event_name(event_t*, string) except +

        bool operator == (CyEventTemplate& ev) except +

        map[string, string] leaves
        map[string, string] branches
        map[string, string] trees

        CyCode* code_link

        event_t event
        meta_t meta


        bool is_event
        bool is_graph
        bool is_selection



