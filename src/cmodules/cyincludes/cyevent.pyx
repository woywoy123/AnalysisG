# distuils: language = c++
# cython: language_level = 3

from cyevent cimport CyEventTemplate
from cytypes cimport meta_t, event_T

from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool
from typing import Union
import pickle

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class EventTemplate:
    cdef CyEventTemplate* ptr
    cdef dict Objects

    def __cinit__(self):
        self.ptr = new CyEventTemplate()
        cdef str x = self.__class__.__name__
        self.ptr.add_eventname(enc(x))
        self.Objects = {}

    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    def __name__(self) -> str: return env(self.ptr.event.event_name)
    def __hash__(self) -> int: return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef EventTemplate o = other
        return self.ptr[0] == o.ptr[0]

    def __getstate__(self) -> event_T:
        cdef event_T x = self.ptr.Export()
        cdef str key
        cdef string pkl
        for key in list(self.__dict__) + ["Objects"]:
            pkl = pickle.dumps(getattr(self, key))
            x.event.pickled_data[enc(key)] = pkl
        return x

    def __setstate__(self, event_T inpt):
        self.ptr.Import(inpt)
        cdef pair[string, string] pkl
        cdef str key
        for pkl in self.ptr.event.pickled_data:
            key = env(pkl.first)
            setattr(self, key, pickle.loads(pkl.second))

    def __getleaves__(self) -> dict:
        cdef str i
        cdef dict leaves = {}
        cdef pair[string, string] x
        for x in self.ptr.leaves: leaves[env(x.first)] = env(x.second)
        for i, v in zip(self.__dict__, self.__dict__.values()):
            if not isinstance(v, str): continue
            leaves[i] = v
        leaves = {"event" : leaves}
        for i in self.Objects:
            try: self.Objects[i] = self.Objects[i]()
            except TypeError: pass
            leaves[i] = self.Objects[i].__getleaves__()

        self.Leaves = []
        for i in leaves: self.Leaves += list(leaves[i].values())
        return leaves

    def __compiler__(self, inpt: Union[dict]):
        cdef dict var_map = self.__getleaves__()
        cdef dict var_leaf = {}
        cdef dict inpt_map = {}
        cdef dict objects = {}
        cdef dict output = {}
        cdef EventTemplate event
        cdef str key, tree, typ_, leaf
        cdef int i
        for tree in set(self.Trees):
            event = self.clone(inpt["MetaData"].__getstate__())
            event.__getleaves__()
            objects.update({"event" : event})
            objects.update(objects["event"].Objects)
            objects["event"].Tree = tree
            output[tree] = {}

            for key in inpt:
                if tree not in key: continue
                if tree not in inpt_map: inpt_map[tree] = {}
                inpt_map[tree][key.split("/")[-1]] = inpt[key]

            if tree not in inpt_map:
                del output[tree]
                del event
                continue

            for typ_ in var_map:
                for key, leaf in var_map[typ_].items():
                    if leaf not in inpt_map[tree]: continue
                    var_leaf[key] = inpt_map[tree][leaf]
                if not len(var_leaf): continue

                if typ_ != "event":
                    objects[typ_].__build__(var_leaf)
                    output[tree][typ_] = objects[typ_].Children
                    continue
                output[tree][typ_] = objects[typ_].__build__(var_leaf)

            try: event = output[tree]["event"]
            except KeyError:
                del output[tree]
                continue
            for key in self.Objects:
                objects = {}
                for i in range(len(output[tree][key])):
                    objects[i] = output[tree][key][i]
                setattr(event, key, objects)
            if event.index == -1: event.index = inpt["EventIndex"]
            else: inpt["MetaData"].eventNumber = event.index
            event.ROOT = inpt["MetaData"].DatasetName + "/" + inpt["MetaData"].DAOD
            event.hash
            output[tree] = event
        return list(output.values())

    def __build__(self, dict variables):
        cdef str keys
        cdef EventTemplate ev = self.clone()
        ev.Tree = self.Tree
        for keys in variables:
            try: variables[keys] = variables[keys].tolist()
            except AttributeError: pass

            try: variables[keys] = variables[keys].pop()
            except AttributeError: pass

            setattr(ev, keys, variables[keys])
        return ev


    def is_self(self, inpt) -> bool:
        if isinstance(inpt, EventTemplate): return True
        return issubclass(inpt.__class__, EventTemplate)

    def clone(self, meta = None) -> EventTemplate:
        v = self.__class__
        v = v()
        if meta is None: return v
        v.ImportMetaData(meta)
        return v

    def CompileEvent(self): pass

    def ImportMetaData(self, meta_t meta):
        self.ptr.ImportMetaData(meta)

    @property
    def index(self) -> int:
        return self.ptr.event.event_index

    @index.setter
    def index(self, val: Union[str, int]):
        try: self.ptr.event.event_index = val
        except TypeError: self.ptr.addleaf(b'index', enc(val))

    @property
    def weight(self) -> double:
        return self.ptr.event.weight

    @weight.setter
    def weight(self, val: Union[str, double]):
        try: self.ptr.event.weight = val
        except TypeError: self.ptr.addleaf(b'weight', enc(val))

    @property
    def deprecated(self) -> bool:
        return self.ptr.event.deprecated

    @deprecated.setter
    def deprecated(self, bool val):
        self.ptr.event.deprecated = val

    @property
    def CommitHash(self) -> str:
        return env(self.ptr.event.commit_hash)

    @CommitHash.setter
    def CommitHash(self, str val):
        self.ptr.event.commit_hash = enc(val)

    @property
    def Tag(self) -> str:
        return env(self.ptr.event.event_tagging)

    @Tag.setter
    def Tag(self, str val):
        self.ptr.event.event_tagging = enc(val)


    @property
    def Tree(self) -> str:
        return env(self.ptr.event.event_tree)

    @Tree.setter
    def Tree(self, str val):
        self.ptr.event.event_tree = enc(val)

    @property
    def Trees(self) -> list:
        cdef pair[string, string] x
        return [env(x.first) for x in self.ptr.trees]

    @Trees.setter
    def Trees(self, val: Union[str, list]):
        cdef str i
        if isinstance(val, str):
            self.ptr.addtree(enc(val), b'Tree')
        elif isinstance(val, list):
            for i in val: self.Trees = i
        else: pass

    @property
    def Branches(self) -> list:
        cdef pair[string, string] x
        return [env(x.first) for x in self.ptr.branches]

    @Branches.setter
    def Branches(self, val: Union[str, list]):
        if isinstance(val, str):
            self.ptr.addbranch(enc(val), b'Branch')
        elif isinstance(val, list):
            for i in val: self.Branches = i
        else: pass

    @property
    def cached(self) -> bool:
        return self.ptr.event.cached

    @cached.setter
    def cached(self, bool val) -> bool:
        self.ptr.event.cached = val

    @property
    def ROOT(self) -> str:
        return env(self.ptr.event.event_root)

    @ROOT.setter
    def ROOT(self, str val):
        self.ptr.event.event_root = enc(val)

    @property
    def hash(self) -> str:
        return env(self.ptr.Hash())

    @property
    def Objects(self) -> dict:
        return self.Objects

    @Objects.setter
    def Objects(self, val):
        if not isinstance(val, dict):
            val = {"title" : val}
        self.Objects = val


