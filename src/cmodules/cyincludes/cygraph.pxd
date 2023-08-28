from cytypes cimport meta_t, graph_t
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../graph/graph.h" namespace "CyTemplate":
    cdef cppclass CyGraphTemplate:
        CyGraphTemplate() except +
        void ImportMetaData(meta_t meta) except +
        void Import(graph_t graph) except +

        string Hash() except +

        graph_t graph
        meta_t meta


