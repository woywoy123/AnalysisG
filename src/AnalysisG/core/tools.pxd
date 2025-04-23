# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool, int, float
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as dref
from libcpp.unordered_map cimport unordered_map

cdef extern from "<tools/tools.h>":

    cdef cppclass tools:
        tools() except+ nogil

        # io.cxx 
        void create_path(string inpt) except+ nogil
        void delete_path(string inpt) except+ nogil
        bool is_file(string inpt) except+ nogil
        void rename(string start, string target) except+ nogil
        string absolute_path(string path) except+ nogil
        vector[string] ls(string path, string ext) except+ nogil

        # strings.cxx
        void replace(string* inpt, string rpl, string rpwl) except+ nogil
        bool has_string(string* inpt, string trg) except+ nogil
        bool ends_with(string* inpt, string val) except+ nogil
        bool has_value(vector[string]* data, string trg) except+ nogil

        vector[string] split(string ipt, string delm) except+ nogil
        vector[string] split(string ipt, int n) except+ nogil
        string hash(string inpt, int l) except+ nogil
        float max(vector[float]*) except+ nogil
        float min(vector[float]*) except+ nogil
        float sum(vector[float]*) except+ nogil

        string encode64(string* data) except+ nogil
        string decode64(string* data) except+ nogil

        vector[vector[int   ]] discretize(vector[int   ]* v, int N) except+ nogil
        vector[vector[float ]] discretize(vector[float ]* v, int N) except+ nogil
        vector[vector[double]] discretize(vector[double]* v, int N) except+ nogil
        vector[vector[bool  ]] discretize(vector[bool  ]* v, int N) except+ nogil
        vector[vector[string]] discretize(vector[string]* v, int N) except+ nogil


ctypedef fused base_types:
    int
    float
    double
    bool
    string

ctypedef fused prim_types:
    int
    float
    double
    bool
    string

cdef extern from "<tools/merge_cast.h>" nogil:
    void merge_data(map[prim_types, vector[base_types]]* out, map[prim_types, vector[base_types]]* p2)
    void merge_data(map[prim_types, base_types]*         out, map[prim_types, base_types]*         p2)
    void merge_data(map[prim_types, base_types]*         out, map[prim_types, base_types]*         p2)

    void merge_data(vector[base_types]* out, vector[base_types]*  p2)

    void merge_data(base_types*  out, base_types*  p2)

    void sum_data(vector[base_types]* out, vector[base_types]* p2)

    void sum_data(base_types*    out, base_types*  p2)

  
cdef string enc(str val)
cdef str env(string val)
cdef list env_vec(vector[string]* inpt)
cdef vector[string] enc_list(list inpt)
cdef list as_list(vector[base_types]* inp)

cdef dict as_dict(map[string, vector[base_types]]* inpt)
cdef void as_map(dict inpt, map[string, base_types]* out)
cdef void as_umap(dict inpt, unordered_map[string, base_types]* out)

cdef dict as_dict_dict(map[string, map[string, vector[base_types]]]* inpt)
cdef dict as_dict_dict_dict(map[string, map[string, map[string, vector[base_types]]]]* inpt)

cdef dict as_basic_dict(map[string, base_types]* inpt)
cdef dict as_basic_udict(unordered_map[string, base_types]* inpt)

cdef dict as_basic_dict_dict(map[string, map[string, base_types]]* inpt)
cdef dict as_basic_dict_dict_f(map[string, map[float, base_types]]* inpt)

cdef class Tools:
    cdef tools* ptr
