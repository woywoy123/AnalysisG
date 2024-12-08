# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool, int
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as dref

cdef extern from "<tools/tools.h>":

    cdef cppclass tools:
        tools() except+ nogil

        # io.cxx 
        void create_path(string inpt) except+ nogil
        void delete_path(string inpt) except+ nogil
        bool is_file(string inpt) except+ nogil
        string absolute_path(string path) except+ nogil
        vector[string] ls(string path, string ext) except+ nogil

        # strings.cxx
        string hash(string inpt, int l) except+ nogil
        bool has_string(string* inpt, string trg) except+ nogil
        void replace(string* inpt, string rpl, string rpwl) except+ nogil
        vector[string] split(string ipt, string delm) except+ nogil
        vector[string] split(string ipt, int n) except+ nogil

        string encode64(string* data) except+ nogil
        string decode64(string* data) except+ nogil



cdef inline string enc(str val): return val.encode("utf-8")
cdef inline str env(string val): return val.decode("utf-8")
cdef inline list env_vec(vector[string]* inpt):
    cdef int i
    cdef list out = [env(inpt.at(i)) for i in range(inpt.size())]
    return out

cdef inline vector[string] enc_list(list inpt):
    cdef str i
    cdef vector[string] out = []
    for i in inpt: out.push_back(enc(i))
    return out

ctypedef fused base_types:
    int
    float
    double
    bool

cdef inline list as_list(vector[base_types]* inp): return list(dref(inp))

cdef inline dict as_basic_dict(map[string, base_types]* inpt):
    cdef dict output = {}
    cdef pair[string, base_types] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

cdef inline dict as_basic_dict_dict_f(map[string, map[float, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[float, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

cdef inline dict as_basic_dict_dict(map[string, map[string, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_basic_dict(&pair.second)
    return output

cdef inline dict as_dict(map[string, vector[base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, vector[base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_list(&pair.second)
    return output

cdef inline dict as_dict_dict(map[string, map[string, vector[base_types]]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, vector[base_types]]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_dict(&pair.second)
    return output

cdef inline dict as_dict_dict_dict(map[string, map[string, map[string, vector[base_types]]]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, map[string, vector[base_types]]]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_dict_dict(&pair.second)
    return output

cdef inline void as_map(dict inpt, map[string, base_types]* out):
    cdef str key
    for key in inpt: dref(out)[enc(key)] = inpt[key]
