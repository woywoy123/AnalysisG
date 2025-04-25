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

cdef extern from "<tools/merge_cast.h>" nogil:
    cdef void merge_data(map[string, vector[int   ]]* out, map[string, vector[int   ]]* p2)
    cdef void merge_data(map[string, vector[float ]]* out, map[string, vector[float ]]* p2)
    cdef void merge_data(map[string, vector[double]]* out, map[string, vector[double]]* p2)
    cdef void merge_data(map[string, vector[bool  ]]* out, map[string, vector[bool  ]]* p2)

    cdef void merge_data(map[int, vector[int   ]]* out, map[int, vector[int   ]]* p2)
    cdef void merge_data(map[int, vector[float ]]* out, map[int, vector[float ]]* p2)
    cdef void merge_data(map[int, vector[double]]* out, map[int, vector[double]]* p2)
    cdef void merge_data(map[int, vector[bool  ]]* out, map[int, vector[bool  ]]* p2)

    cdef void merge_data(map[double, vector[int   ]]* out, map[double, vector[int   ]]* p2)
    cdef void merge_data(map[double, vector[float ]]* out, map[double, vector[float ]]* p2)
    cdef void merge_data(map[double, vector[double]]* out, map[double, vector[double]]* p2)
    cdef void merge_data(map[double, vector[bool  ]]* out, map[double, vector[bool  ]]* p2)

    cdef void merge_data(map[bool, vector[int   ]]* out, map[bool, vector[int   ]]* p2)
    cdef void merge_data(map[bool, vector[float ]]* out, map[bool, vector[float ]]* p2)
    cdef void merge_data(map[bool, vector[double]]* out, map[bool, vector[double]]* p2)
    cdef void merge_data(map[bool, vector[bool  ]]* out, map[bool, vector[bool  ]]* p2)

    cdef void merge_data(map[string, int   ]* out, map[string, int   ]* p2)
    cdef void merge_data(map[string, float ]* out, map[string, float ]* p2)
    cdef void merge_data(map[string, double]* out, map[string, double]* p2)
    cdef void merge_data(map[string, bool  ]* out, map[string, bool  ]* p2)

    cdef void merge_data(map[int, int   ]* out, map[int, int   ]* p2)
    cdef void merge_data(map[int, float ]* out, map[int, float ]* p2)
    cdef void merge_data(map[int, double]* out, map[int, double]* p2)
    cdef void merge_data(map[int, bool  ]* out, map[int, bool  ]* p2)

    cdef void merge_data(map[double, int   ]* out, map[double, int   ]* p2)
    cdef void merge_data(map[double, float ]* out, map[double, float ]* p2)
    cdef void merge_data(map[double, double]* out, map[double, double]* p2)
    cdef void merge_data(map[double, bool  ]* out, map[double, bool  ]* p2)

    cdef void merge_data(map[bool, int   ]* out, map[bool, int   ]* p2)
    cdef void merge_data(map[bool, float ]* out, map[bool, float ]* p2)
    cdef void merge_data(map[bool, double]* out, map[bool, double]* p2)
    cdef void merge_data(map[bool, bool  ]* out, map[bool, bool  ]* p2)

    cdef void merge_data(vector[int   ]* out, vector[int   ]*  p2)
    cdef void merge_data(vector[float ]* out, vector[float ]*  p2)
    cdef void merge_data(vector[double]* out, vector[double]*  p2)
    cdef void merge_data(vector[bool  ]* out, vector[bool  ]*  p2)

    cdef void sum_data(vector[int   ]* out, vector[int   ]* p2)
    cdef void sum_data(vector[float ]* out, vector[float ]* p2)
    cdef void sum_data(vector[double]* out, vector[double]* p2)
    cdef void sum_data(vector[bool  ]* out, vector[bool  ]* p2)

    cdef void merge_data(int   *  out, int   *  p2)
    cdef void merge_data(float *  out, float *  p2)
    cdef void merge_data(double*  out, double*  p2)
    cdef void merge_data(bool  *  out, bool  *  p2)



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
