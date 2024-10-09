# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string
from libcpp.map cimport map, pair
from libcpp.vector cimport vector
from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *

cdef class SelectionTemplate:
    def __cinit__(self):
        if type(self) is not SelectionTemplate: return
        self.ptr = new selection_template()

    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue
        self.ptr.passed_weights = <map[string, map[string, float]]>(inpt["weights"])
        self.ptr.matched_meta   = <map[string, meta_t]>(inpt["meta"])

    def __dealloc__(self):
        if type(self) is not SelectionTemplate: return
        del self.ptr

    def __hash__(self):
        return int(string(self.ptr.hash).substr(0, 8), 0)

    def __reduce__(self):
        cdef list keys = self.__dir__()
        cdef dict out = {}
        out["data"] = {i : getattr(self, i) for i in keys if not i.startswith("__")}
        out["weights"] = self.ptr.passed_weights
        out["meta"] = self.ptr.matched_meta
        return self.__class__, (out,)

    cdef void transform_dict_keys(self): pass

    @property
    def PassedWeights(self): return as_basic_dict_dict(&self.ptr.passed_weights)

    @property
    def GetMetaData(self):
        cdef Meta data
        cdef dict out = {}
        cdef pair[string, meta_t] itm
        for itm in self.ptr.matched_meta:
            data = Meta()
            data.ptr.meta_data = itm.second
            out[env(itm.first)] = data
        return out
