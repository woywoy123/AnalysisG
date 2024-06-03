# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "<tools/tools.h>":

    cdef cppclass tools:
        tools() except +

        # io.cxx 
        void create_path(string inpt) except +
        void delete_path(string inpt) except +
        bool is_file(string inpt) except +

        # strings.cxx
        string hash(string inpt, int l) except +
        bool has_string(string* inpt, string trg) except +
        void replace(string* inpt, string rpl, string rpwl) except +
        vector[string] split(string ipt, string delm) except +
        vector[string] split(string ipt, int n) except +

cdef inline string enc(str val): return val.encode("utf-8")
cdef inline str env(string val): return val.decode("utf-8")
cdef inline list env_vec(vector[string]* inpt):
    cdef int i
    cdef list out = []
    for i in inpt.size(): out.append(env(inpt.at(i)))
    return out

cdef inline vector[string] enc_list(list inpt):
    cdef int i
    cdef vector[string] out = []
    for i in inpt.size(): out.push_back(enc(inpt[i]))
    return out
