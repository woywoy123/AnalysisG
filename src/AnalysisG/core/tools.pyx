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
    for key in inpt: dref(out)[enc(key)] = inpt[key] if not isinstance(inpt[key], str) else enc(inpt[key])

cdef void as_umap(dict inpt, unordered_map[string, base_types]* out):
    cdef str key
    for key in inpt: dref(out)[enc(key)] = inpt[key] if not isinstance(inpt[key], str) else enc(inpt[key])

cdef dict as_basic_dict_dict_f(map[string, map[float, base_types]]* inpt):
    cdef dict output = {}
    cdef pair[string, map[float, base_types]] itr
    for pair in dref(inpt): output[env(pair.first)] = pair.second
    return output

cdef class Tools:
    def __cinit__(self): self.ptr = new tools()
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    def create_path(self, str pth): self.ptr.create_path(enc(pth))
    def delete_path(self, str pth): self.ptr.delete_path(enc(pth))
    def is_file(self, str pth): return self.ptr.is_file(enc(pth))
    def rename(self, str src, str dst): self.ptr.rename(enc(src), enc(dst))
    def abs(self, str pth): return env(self.ptr.absolute_path(enc(pth)))

    def ls(self, str pth, str ext):
        cdef vector[string] pd = self.ptr.ls(enc(pth), enc(ext))
        return env_vec(&pd)

    def replace(self, str val, str rpl, str rpwl):
        cdef string sx = enc(val)
        self.ptr.replace(&sx, enc(rpl), enc(rpwl))
        return env(sx)

    def has_substring(self, str val, str rpl):
        cdef string sx = enc(val)
        return self.ptr.has_string(&sx, enc(rpl))

    def ends_with(self, str val, str rpl):
        cdef string sx = enc(val)
        return self.ptr.ends_with(&sx, enc(rpl))

    def has_value(self, list data, str trg):
        cdef vector[string] pd = enc_list(data)
        return self.ptr.has_value(&pd, enc(trg))

    def split(self, str data, trg):
        cdef vector[string] pd = []
        if   isinstance(trg, int): pd = self.ptr.split(enc(data), <int>(trg))
        elif isinstance(trg, str): pd = self.ptr.split(enc(data), enc(trg))
        return as_list(&pd)

    def hash(self, str data, int lx = 8): return env(self.ptr.hash(enc(data), lx))
    def encode64(self, str data): 
        cdef string v = enc(data)
        return env(self.ptr.encode64(&v))

    def decode64(self, str data): 
        cdef string v = enc(data)
        return env(self.ptr.decode64(&v))

    def discretize(self, list data, int lx): 
        if not len(data): return []
        cdef vector[string] v = []
        cdef vector[int] vi = []
        if isinstance(data[0], str): 
            v = enc_list(data)
            return self.ptr.discretize(&v, lx)
        elif isinstance(data[0], int):
            vi = <vector[int]>(data)
            return self.ptr.discretize(&vi, lx)
        return []


