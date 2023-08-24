from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

from cytypes cimport code_t

cdef extern from "../code/code.h" namespace "Code":
    cdef cppclass CyCode:
        CyCode() except +

        void Hash() except +
        void ImportCode(code_t) except +
        code_t ExportCode() except +

        string hash
        code_t container

        bool operator==(CyCode* code) except +

