# distutils: language=c++
# cython: language_level=3
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

import pickle
from typing import Union
from AnalysisG.core.particle_template cimport particle_template
from AnalysisG.core.structs cimport particle_t
from AnalysisG.core.tools cimport env, enc

cdef class ParticleTemplate:

    def __cinit__(self):
        self.children = []
        self.parents = []
        if type(self) is not ParticleTemplate: return
        self.ptr = new particle_template()

    def __init__(self): pass
    def __dealloc__(self):
        if type(self) is not ParticleTemplate: return
        del self.ptr

    def __hash__(self): return int(string(self.ptr.hash).substr(0, 8), 0)
    def __add__(self, ParticleTemplate other):
        cdef ParticleTemplate p = self.clone()
        p.ptr.iadd(other.ptr)
        p.ptr.iadd(self.ptr)
        if len(other.Children): p.Children += other.Children
        if len(other.Parent):   p.Parent   += other.Parent
        if len(self.Children):  p.Children += self.Children
        if len(self.Parent):    p.Parent   += self.Parent
        return p

    def __radd__(self, other):
        if not self.is_self(other): return self
        return self.__add__(other)

    def __eq__(self, other):
        if not self.is_self(other): return False
        cdef ParticleTemplate o = other
        return self.ptr[0] == o.ptr[0]

    def __getleaves__(self):
        cdef str i
        cdef dict leaves = {}
        cdef pair[string, string] x
        for x in self.ptr.leaves: leaves[env(x.first)] = env(x.second)
        for i, v in zip(self.__dict__, self.__dict__.values()):
            if not isinstance(v, str): continue
            self.ptr.leaves[enc(i)] = enc(v)
            leaves[i] = v
        return leaves


#     cdef bool _pkld
#     cdef particle_t data
# 
#     def __cinit__(self):
#         self.ptr = new CyParticleTemplate()
#         self.data = self.ptr.data
#         self.Children = []
#         self.Parent = []
#         self._pkld = False
# 
#     def __getdata__(self) -> tuple:
#         if self._pkld: return ({}, self.data)
#         self._pkld = True
#         self.data = self.ptr.Export()
# 
#         cdef str key
#         cdef list get = list(self.__dict__)
#         cdef dict pkl = { key : self.__dict__[key] for key in get}
#         pkl["Children"] = self.Children
#         pkl["Parent"] = self.Parent
#         return (pkl, self.data)
# 
#     def __setdata__(self, tuple inpt):
#         cdef str key
#         self.ptr.Import(inpt[1])
#         for key in inpt[0]: setattr(self, key, inpt[0][key])
# 
 
#     def __build__(self, dict variables):
#         cdef ParticleTemplate p
#         cdef dict inpt = {}
#         cdef dict x = {}
#         cdef bool get
#         cdef str k
#         cdef dict lv, p_
# 
#         cdef list keys = list(self.__getleaves__())
#         for k in variables:
#             if k.split("/")[-1] not in keys: continue
#             try: inpt[k] = variables[k].tolist()
#             except AttributeError:
#                 if isinstance(variables[k], str): continue
#                 inpt[k] = variables[k]
# 
#         if not len(inpt): return {}
#         while True:
#             x = {}
#             get = False
#             for k in list(inpt):
#                 try: inpt[k] = inpt[k].tolist()
#                 except AttributeError: pass
# 
#                 try: x[k] = inpt[k].pop()
#                 except AttributeError: x[k] = inpt[k]
#                 except IndexError: pass
# 
#                 if k not in x: continue
# 
#                 try: x[k] = x[k].item()
#                 except AttributeError: pass
# 
#                 try: len(x[k])
#                 except TypeError: get = True
# 
#             if len(x) == 0: break
#             if not get: self.__build__(x)
#             else:
#                 p = self.clone()
#                 p_ = {k.split("/")[-1] : x[k] for k in x}
#                 for k in p_:
#                     try: setattr(p, k, p_[k])
#                     except KeyError: pass
#                 self.Children.append(p)
#         return {i : obj for i, obj in enumerate(self.Children)}
# 
    cpdef ParticleTemplate clone(self):
        v = self.__class__
        v = v()
        v.Type = self.Type
        return v

    def is_self(self, inpt):
        if isinstance(inpt, ParticleTemplate): return True
        return issubclass(inpt.__class__, ParticleTemplate)

    cpdef double DeltaR(self, ParticleTemplate other):
        return self.ptr.DeltaR(other.ptr)

    @property
    def hash(self) -> str: return env(self.ptr.hash)

    @property
    def index(self) -> int: return self.ptr.data.index

    @property
    def px(self) -> double: return self.ptr.px

    @property
    def py(self) -> double: return self.ptr.py

    @property
    def pz(self) -> double: return self.ptr.pz

    @property
    def pt(self) -> double: return self.ptr.pt

    @property
    def eta(self) -> double: return self.ptr.eta

    @property
    def phi(self) -> double: return self.ptr.phi

    @property
    def e(self) -> double: return self.ptr.e

    @property
    def Mass(self) -> double: return self.ptr.mass

    @property
    def charge(self) -> double: return self.ptr.charge

    @property
    def pdgid(self) -> int: return self.ptr.pdgid

    @property
    def symbol(self) -> str : return env(self.ptr.symbol)

    @property
    def Type(self) -> str: return env(self.ptr.data.type)

    @property
    def lepdef(self) -> vector[int]: return self.ptr.data.lepdef

    @property
    def nudef(self) -> vector[int]: return self.ptr.data.nudef

    @property
    def is_lep(self) -> bool: return self.ptr.is_lep

    @property
    def is_nu(self) -> bool: return self.ptr.is_nu

    @property
    def is_b(self) -> bool: return self.ptr.is_b

    @property
    def is_add(self) -> bool: return self.ptr.is_add

    @property
    def LeptonicDecay(self) -> bool: return self.ptr.lep_decay

    @index.setter
    def index(self, val: Union[str, int, float, list]):
        try: self.ptr.data.index = val
        except TypeError:
            try: self.ptr.add_leaf(b'index', enc(val))
            except: self.__dict__["index"] = val

    @px.setter
    def px(self, val: Union[str, double]):
        try: self.ptr.px = val
        except TypeError: self.ptr.add_leaf(b'px', enc(val))

    @py.setter
    def py(self, val: Union[str, double]):
        try: self.ptr.py = val
        except TypeError: self.ptr.add_leaf(b'py', enc(val))

    @pz.setter
    def pz(self, val: Union[str, double]):
        try: self.ptr.pz = val
        except TypeError: self.ptr.add_leaf(b'pz', enc(val))

    @pt.setter
    def pt(self, val: Union[str, double]):
        try: self.ptr.pt = val
        except TypeError: self.ptr.add_leaf(b'pt', enc(val))

    @eta.setter
    def eta(self, val: Union[str, double]):
        try: self.ptr.eta = val
        except TypeError: self.ptr.add_leaf(b'eta', enc(val))

    @phi.setter
    def phi(self, val: Union[str, double]):
        try: self.ptr.phi = val
        except TypeError: self.ptr.add_leaf(b'phi', enc(val))

    @e.setter
    def e(self, val: Union[str, double]):
        try: self.ptr.e = val
        except TypeError: self.ptr.add_leaf(b'e', enc(val))

    @Mass.setter
    def Mass(self, val: Union[str, double]):
        try: self.ptr.mass = val
        except TypeError: self.ptr.add_leaf(b'Mass', enc(val))

    @charge.setter
    def charge(self, val: Union[str, double]):
        try: self.ptr.charge = val
        except TypeError: self.ptr.add_leaf(b'charge', enc(val))

    @pdgid.setter
    def pdgid(self, val: Union[str, double, int]):
        try: self.ptr.pdgid = val
        except TypeError: self.ptr.add_leaf(b'pdgid', enc(val))

    @symbol.setter
    def symbol(self, str val): self.ptr.symbol = enc(val)

    @Type.setter
    def Type(self, str val): self.ptr.data.type = enc(val)

    @lepdef.setter
    def lepdef(self, vector[int] val): self.ptr.data.lepdef = val

    @nudef.setter
    def nudef(self, vector[int] val): self.ptr.data.nudef = val

    @property
    def Children(self) -> list:
        self.children = list(set(self.children))
        return self.children

    @Children.setter
    def Children(self, inpt):
        cdef ParticleTemplate p
        cdef particle_template* ptx

        if not len(inpt):
            self.ptr.children.clear()
            self.children = []
            return

        if isinstance(inpt, list):
            for p in inpt:
                ptx = p.ptr
                if not self.ptr.register_child(ptx): continue
                self.children.append(p)
        else: self.Children += [inpt]

    @property
    def Parent(self) -> list:
        self.parents = list(set(self.parents))
        return self.parents

    @Parent.setter
    def Parent(self, inpt):
        cdef ParticleTemplate p
        cdef particle_template* ptx

        if not len(inpt):
            self.ptr.parents.clear();
            self.parents = []

        if isinstance(inpt, list):
            for p in inpt:
                ptx = p.ptr
                if not self.ptr.register_parent(ptx): continue
                self.parents.append(p)
        else: self.Parent += [inpt]



