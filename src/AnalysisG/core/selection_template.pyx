# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string
from libcpp.map cimport map, pair
from libcpp.vector cimport vector

from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *
from AnalysisG.core.io import IO

import pathlib
import pickle

cdef class SelectionTemplate:
    def __cinit__(self):
        if type(self) is not SelectionTemplate: return
        self.root_leaves = {}
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

    def __name__(self): return env(self.ptr.name)

    def __reduce__(self):
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        cdef dict out = {}
        out["data"] = {i : getattr(self, i) for i in keys if not callable(getattr(self, i))}
        out["weights"] = self.ptr.passed_weights
        out["meta"]    = self.ptr.matched_meta
        return self.__class__, (out,)

    cdef void transform_dict_keys(self): pass

    def dump(self, str path = "./pkl-data", str name = ""):
        if not len(name): name = env(self.ptr.name)
        pathlib.Path(path).mkdir(parents = True, exist_ok = True)
        try: pickle.dump(self, open(path + "/" + name + ".pkl", "wb"))
        except OSError: print("Failed to save the SelectionTemplate")

    def load(self, str path = "./pkl-data", str name = ""):
        if not len(name): name = env(self.ptr.name)
        try: return pickle.load(open(path + "/" + name + ".pkl", "rb"))
        except OSError: print("Failed to load the SelectionTemplate")
        except EOFError: print("Failed to load the SelectionTemplate")
        return None

    @property
    def PassedWeights(self): return as_basic_dict_dict(&self.ptr.passed_weights)

    def HashToWeightFile(self, hash_):
        cdef str hash
        cdef vector[string] hashes = []
        if isinstance(hash_, list):   hashes = [enc(hash) for hash in hash_]
        elif isinstance(hash_, dict): hashes = [enc(hash) for hash in hash_]
        else: hashes = [enc(hash_)]
        cdef vector[map[string, float]] rev = self.ptr.reverse_hash(&hashes)

        cdef map[string, float] i
        return [tuple(dict(i).items())[0] for i in rev]

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

    def Postprocessing(self): pass

    def InterpretROOT(self, str path, str tree):
        if self.root_leaves is None or not len(self.root_leaves):
            print("Failed to interpret!")
            print("Please set the attribute ( dictionary {<leaves> : <fx(class, data)>} ): 'root_leaves'")
            return self

        cdef string kx, ky
        cdef list rn = []
        cdef list li = list(self.root_leaves)
        cdef list lo = list(self.root_leaves.values())
        cdef vector[string] lx = <vector[string]>([enc(k) for k in li])
        cdef vector[string] lf = []
        cdef int lk, lt

        cdef dict i
        cdef bool trig = False

        cdef pair[int, int] pi
        cdef pair[int, string] px

        cdef map[int, string] associate = {}
        cdef map[int, int] idx = {}; 

        cdef tools tl = tools()

        iox = IO()
        iox.Files = path
        iox.Trees = [tree]
        iox.Leaves = li
        iox.Verbose = False
        for i in iox:
            li = list(i.values())
            for pi in idx: lo[pi.second](self, (rn[pi.second], li[pi.first]))            
            if trig: continue

            li = list(i)
            for lk in range(len(li)):
                try: kx = enc(li[lk])
                except TypeError: kx = li[lk]
                lf = [ky for ky in lx if tl.ends_with(&kx, ky)]
                if not lf.size(): continue
                associate[lk] = lf[0]

            for lt in range(lx.size()):
                lk = -1
                ky = lx[lt]
                for px in associate: 
                    if px.second != ky: continue
                    lk = px.first
                    break
                if lk == -1: continue
                idx[lk] = lt
            trig = True
            li = list(i.values())
            rn = list(self.root_leaves)
            for pi in idx: lo[pi.second](self, (rn[pi.second], li[pi.first]))
        self.Postprocessing()
        return self
