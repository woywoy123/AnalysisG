from cytypes cimport meta_t, selection_t
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../selection/selection.h" namespace "CyTemplate":
    cdef cppclass CySelectionTemplate:
        CySelectionTemplate() except +
        void ImportMetaData(meta_t meta) except +
        void Import(selection_t graph) except +

        string Hash() except +

        selection_t graph
        meta_t meta


