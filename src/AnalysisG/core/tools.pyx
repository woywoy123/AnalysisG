# distutils: language=c++
# cython: language_level=3
from libcpp cimport bool, int
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as dref
from libcpp.unordered_map cimport unordered_map


cdef string enc(str val): return val.encode("utf-8")
cdef str env(string val): return val.decode("utf-8")
cdef list env_vec(vector[string]* inpt):
    cdef int i
    cdef list out = [env(inpt.at(i)) for i in range(inpt.size())]
    return out

cdef vector[string] enc_list(list inpt):
    cdef str i
    cdef vector[string] out = []
    for i in inpt: out.push_back(enc(i))
    return out

cdef dict as_basic_dist_dict_f(map[string, map[float, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[float, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

cdef list as_list(vector[base_types]* inp): return list(dref(inp))

cdef dict as_basic_dict(map[string, base_types]* inpt):
    cdef dict output = {}
    cdef pair[string, base_types] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

cdef dict as_basic_udict(unordered_map[string, base_types]* inpt):
    cdef dict output = {}
    cdef pair[string, base_types] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output


cdef dict as_basic_dict_dict(map[string, map[string, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_basic_dict(&pair.second)
    return output

cdef dict as_dict(map[string, vector[base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, vector[base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_list(&pair.second)
    return output

cdef dict as_dict_dict(map[string, map[string, vector[base_types]]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, vector[base_types]]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_dict(&pair.second)
    return output

cdef dict as_dict_dict_dict(map[string, map[string, map[string, vector[base_types]]]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[string, map[string, vector[base_types]]]] itr
    for pair in dref(inpt): output[env(pair.first)] = as_dict_dict(&pair.second)
    return output

cdef void as_map(dict inpt, map[string, base_types]* out):
    cdef str key
    for key in inpt: dref(out)[enc(key)] = inpt[key]

cdef void as_umap(dict inpt, unordered_map[string, base_types]* out):
    cdef str key
    for key in inpt: dref(out)[enc(key)] = inpt[key]

cdef dict as_basic_dict_dict_f(map[string, map[float, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[float, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

