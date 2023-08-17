# distuils: language = c++
# cython: language_level = 3
from cyevent cimport CyEventTemplate
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool
from typing import Union

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class EventTemplate:
    cdef CyEventTemplate* ptr
    cdef public dict Objects

    def __cinit__(self):
        self.ptr = new CyEventTemplate()

    def __init__(self):
        self.ptr.event_name = enc(self.__class__.__name__)

    def __dealloc__(self):
        del self.ptr

    def __name__(self) -> str:
        return env(self.ptr.event_name)

    def __hash__(self) -> int:
        return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef EventTemplate o = other
        return self.ptr == o.ptr

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

        for tree in self.Trees:
            event = self.clone()
            event.__getleaves__()
            objects.update({"event" : event})
            objects.update(objects["event"].Objects)
            if tree not in inpt_map: inpt_map[tree] = {}
            for key in inpt:
                if tree not in key: continue
                inpt_map[tree][key.split("/")[-1]] = inpt[key]

            objects["event"].Tree = tree
            output[tree] = {}
            for typ_ in var_map:
                var_leaf = var_map[typ_]
                var_leaf = {
                        key : inpt_map[tree][leaf] 
                        for (key, leaf) in var_leaf.items()
                        if leaf in inpt_map[tree]
                }
                if len(var_leaf) == 0: continue
                if typ_ == "event":
                    output[tree][typ_] = objects[typ_].__build__(var_leaf)
                else:
                    objects[typ_].__build__(var_leaf)
                    output[tree][typ_] = objects[typ_].Children
            if "event" not in output[tree]: 
                del event
                del output[tree]
                continue
            event = output[tree]["event"]
            for key in self.Objects:
                if key not in output[tree]: continue
                setattr(event, key, {
                    i : output[tree][key][i] for i in range(len(output[tree][key]))
                })
            output[tree] = event

            if event.index == -1: event.index = inpt["EventIndex"]
            else: inpt["MetaData"].event_index = event.index
            event.hash = inpt["MetaData"].DatasetName + "/" + inpt["MetaData"].DAOD
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
        return issubclass(inpt.__class__, EventTemplate)

    def clone(self) -> EventTemplate:
        v = self.__new__(self.__class__)
        v.__init__()
        return v

    def CompileEvent(self): pass

    @property
    def hash(self) -> str: return env(self.ptr.Hash())

    @hash.setter
    def hash(self, str val): self.ptr.Hash(enc(val))


    @property
    def index(self) -> int: return self.ptr.event_index

    @index.setter
    def index(self, val: Union[str, int]):
        try: self.ptr.event_index = val
        except TypeError: self.ptr.addleaf(b'index', enc(val))


    @property
    def weight(self) -> double: return self.ptr.weight

    @weight.setter
    def weight(self, val: Union[str, double]):
        try: self.ptr.weight = val
        except TypeError: self.ptr.addleaf(b'weight', enc(val))


    @property
    def deprecated(self) -> bool: return self.ptr.deprecated

    @deprecated.setter
    def deprecated(self, bool val): self.ptr.deprecated = val


    @property
    def CommitHash(self) -> str: return env(self.ptr.commit_hash)

    @CommitHash.setter
    def CommitHash(self, str val):
        self.ptr.commit_hash = enc(val)


    @property
    def Tag(self) -> str: return env(self.ptr.event_tagging)

    @Tag.setter
    def Tag(self, str val): self.ptr.event_tagging = enc(val)


    @property
    def Tree(self) -> str: return env(self.ptr.event_tree)

    @Tree.setter
    def Tree(self, str val): self.ptr.event_tree = enc(val)

    @property
    def Trees(self) -> list:
        cdef pair[string, string] x
        return [env(x.first) for x in self.ptr.trees]

    @Trees.setter
    def Trees(self, val: Union[str, list]):
        cdef str i
        if isinstance(val, str):
            self.ptr.trees[enc(val)] = enc("Tree")
        elif isinstance(val, list):
            for i in val: self.ptr.trees[enc(i)] = enc(i)
        else: pass

    @property
    def Branches(self) -> list:
        cdef pair[string, string] x
        return [env(x.first) for x in self.ptr.branches]

    @Branches.setter
    def Branches(self, val: Union[str, list]):
        if isinstance(val, str):
            self.ptr.branches[enc(val)] = enc("Branches")
        elif isinstance(val, list):
            for i in val: self.ptr.branches[enc(i)] = enc(i)
        else: pass

    @property
    def cached(self) -> bool: return self.ptr.cached


