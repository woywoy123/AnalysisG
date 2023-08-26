from cytypes cimport meta_t, event_t, graph_t, selection_t
from cytypes cimport event_T
from cycode cimport CyCode

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

        bool operator == (CyEventTemplate& ev) except +

        map[string, string] leaves
        map[string, string] branches
        map[string, string] trees

        map[string, CyCode*] this_code
        event_t event
        meta_t meta



    cdef cppclass CyGraphTemplate:
        CyGraphTemplate() except +
        void ImportMetaData(meta_t meta) except +
        void Import(graph_t graph) except +

        string Hash() except +

        graph_t graph
        meta_t meta




    cdef cppclass CySelectionTemplate:
        CySelectionTemplate() except +
        void ImportMetaData(meta_t meta) except +
        void Import(selection_t selection) except +

        string Hash() except +

        selection_t selection
        meta_t meta
