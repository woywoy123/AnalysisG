from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../code/code.h":
    struct ExportCode:
        vector[string] input_params
        vector[string] co_vars

        map[string, string] param_space

        string function_name
        string class_name
        string hash
        string source_code
        string object_code

        bool is_class
        bool is_function
        bool is_callable
        bool is_initialized
        bool has_param_variable

cdef extern from "../code/code.h" namespace "Code":
    cdef cppclass CyCode:
        CyCode() except +
        void Hash() except +
        bool operator==(CyCode* code) except +
        ExportCode MakeMapping() except +
        void ImportCode(ExportCode) except +

        vector[string] input_params
        vector[string] co_vars

        map[string, string] param_space

        string function_name
        string class_name
        string hash

        string source_code
        string object_code

        bool is_class
        bool is_function

        bool is_callable
        bool is_initialized

        bool has_param_variable

        map[string, ExportCode] dependency
