# distuils: language = c++
# cython: language_level = 3
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp.vector cimport vector
from libcpp cimport bool

from cycode cimport CyCode
from cytypes cimport code_t

import inspect
import os, sys
import pickle

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Code:

    cdef CyCode* ptr
    cdef public _x
    cdef fx

    def __cinit__(self):
        self.ptr = new CyCode()
        self.fx = None

    def __init__(self, instance = None):
        if instance is None: return
        self._x = instance
        self.__getbasic__()

    def __dealloc__(self): del self.ptr
    def __hash__(self) -> int: return int(self.hash[:8], 0)
    def __getstate__(self) -> code_t: return self.ptr.ExportCode()
    def __setstate__(self, code_t inpt): self.ptr.ImportCode(inpt)
    def __getfunction__(self): self.__getinputs__(self._x)
    def is_self(self, inpt) -> bool: return isinstance(inpt, Code)
    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef Code o = other
        return self.ptr[0]==o.ptr[0]

    def __getbasic__(self):
        cdef str name
        cdef list types
        types = list(type(self._x).__dict__)
        if "__init__" in types: self.is_class = True
        else: self.is_function = True
        self.is_callable = callable(self._x)

        try:
            name = self._x.__qualname__
            self.is_initialized = False
            if name is None: raise AttributeError
        except AttributeError:
            self.is_initialized = True
            name = self._x.__class__.__name__

        if self.is_class: self.class_name = name
        else: self.function_name = name

        if self.is_class: self.__getclass__()
        else: self.__getfunction__()

        self.source_code = open(os.path.abspath(
                sys.modules[self._x.__module__].__file__
                ), "r").read()

        t = inspect.getsource
        try: self.object_code = t(self._x)
        except TypeError: self.object_code = t(self._x.__class__)
        self.__trace__()
        self.ptr.Hash()

    def __getclass__(self):
        ac = type(self._x.__init__).__name__
        if ac == "function": self.is_initialized = False
        else: self.is_initialized = True
        self.__getinputs__(self._x.__init__)


    def __trace__(self):
        if self.is_function: return
        cdef str src, dep, tmp, k
        cdef dict inpt
        cdef list out

        def recursive_resolve(inpt, src, start = None):
            if start is None: start = src
            out = []
            if src not in inpt: return out
            for dep in inpt[src]:
                tmp = start + "->" + dep
                for k in recursive_resolve(inpt, dep, tmp):
                    if tmp not in k: continue
                    else: tmp = k; break
                out.append(tmp)
            return out

        cdef list split = self.source_code.split("\n")
        cdef dict class_name = {}
        cdef dict this_import = {}
        cdef list imports_ = []
        cdef str i, name

        for i in split:
            if "class" not in i: continue
            if ":" not in i: continue
            name, dep = i.split("(")
            name = name.lstrip("class)").replace(" ", "")
            out = dep.rstrip("):").split(",")
            class_name[name] = out

        for i in split:
            if "import" not in i: continue
            for k in i.split("import")[-1].split(" "):
                k = k.replace(" ", "").replace(",", "")
                if not len(k): continue
                out.append(k)
            this_import[i] = out
            imports_ += out
            out = []

        inpt = {}
        for i in class_name:
            out = recursive_resolve(class_name, i)
            for k in out:
                dep = k.split("->")[-1]
                inpt[k] = k.split("->")
                inpt[k].reverse()
                if dep in class_name: continue
                if dep in imports_: inpt[k].pop(0)

        for i in inpt:
            self.ptr.container.trace[enc(i)] = [enc(k) for k in inpt[i]]

        for i in this_import:
            self.ptr.container.extern_imports[enc(i)] = [enc(k) for k in this_import[i]]

    def __getinputs__(self, val):
        cdef int start = 0
        if self.is_class: start = 1
        co_vars  = list(val.__code__.co_varnames)[start:]
        defaults = val.__defaults__
        defaults = [] if defaults is None else list(defaults)
        if val.__defaults__ is not None: pass
        elif len(co_vars): pass
        else: return
        self.input_params = co_vars
        defaults.reverse()
        co_vars.reverse()
        self.defaults = defaults
        self.co_vars  = co_vars

    def AddDependency(self, list inpt):
        cdef code_t itr
        cdef map[string, code_t] x = {}
        for itr in inpt: x[itr.hash] = itr
        self.ptr.AddDependency(x)


    def __mergedependency__(self):
        cdef str cls_name, find
        cdef dict req = self.extern_imports
        cdef dict found
        cdef pair[string, CyCode*] itr
        cdef code_t it
        cdef int i

        for find in req:
            found = {}
            for itr in self.ptr.dependency:
                it = itr.second.container
                cls_name = env(it.class_name)
                if cls_name not in req[find]: continue
                found[req[find].index(cls_name)] = env(it.source_code)
            if not len(found): continue
            for i in found: req[find][i] = found[i]
            req[find] = "\n".join(list(set(req[find])))
            self.source_code = self.source_code.replace(find, req[find])

    def __call__(self, *args, **kargs):
        if not self.is_callable: return
        cdef str i
        cdef dict params
        x = self.InstantiateObject
        if len(kargs):
            params = self.input_params
            for i, k in params:
                if i not in kargs: params[i] = k
                else: params[i] = kargs[i]
            return x(**params)
        if len(args): return x(*[k for k in args])
        return x

    @property
    def InstantiateObject(self):
        if self.fx is None:
            self.__mergedependency__()
            exec(self.source_code, globals())
            if self.is_class: fx = globals()[self.class_name]
            else: fx = globals()[self.function_name]
            self.fx = fx
        if not len(self.co_vars): return self.fx()
        return self.fx

    @property
    def hash(self) -> str:
        self.ptr.Hash()
        return env(self.ptr.hash)

    @property
    def fx(self): return self.fx

    @fx.setter
    def fx(self, val): self.fx = val

    @property
    def is_class(self) -> bool: return self.ptr.container.is_class

    @is_class.setter
    def is_class(self, bool val): self.ptr.container.is_class = val

    @property
    def is_function(self) -> bool: return self.ptr.container.is_function

    @is_function.setter
    def is_function(self, bool val): self.ptr.container.is_function = val

    @property
    def is_callable(self) -> bool: return self.ptr.container.is_callable

    @is_callable.setter
    def is_callable(self, bool val): self.ptr.container.is_callable = val

    @property
    def is_initialized(self) -> bool: return self.ptr.container.is_initialized

    @is_initialized.setter
    def is_initialized(self, bool val): self.ptr.container.is_initialized = val

    @property
    def function_name(self) -> str: return env(self.ptr.container.function_name)

    @function_name.setter
    def function_name(self, str val): self.ptr.container.function_name = enc(val)


    @property
    def class_name(self) -> str: return env(self.ptr.container.class_name)

    @class_name.setter
    def class_name(self, str val): self.ptr.container.class_name = enc(val)

    @property
    def source_code(self) -> str: return env(self.ptr.container.source_code)

    @source_code.setter
    def source_code(self, str val): self.ptr.container.source_code = enc(val)

    @property
    def object_code(self) -> str: return env(self.ptr.container.object_code)

    @object_code.setter
    def object_code(self, str val): self.ptr.container.object_code = enc(val)

    @property
    def co_vars(self) -> list: return [env(i) for i in self.ptr.container.co_vars]

    @co_vars.setter
    def co_vars(self, list val): self.ptr.container.co_vars = [enc(i) for i in val]

    @property
    def input_params(self) -> dict:
        cdef dict out = {}
        cdef str i, j
        for i, j in zip(self.co_vars, self.defaults): out[i] = j
        return out

    @input_params.setter
    def input_params(self, list val): self.ptr.container.input_params = [enc(i) for i in val]

    @property
    def defaults(self) -> list:
        try: return pickle.loads(self.ptr.container.defaults)
        except EOFError: return []

    @defaults.setter
    def defaults(self, list val): self.ptr.container.defaults = pickle.dumps(val)

    @property
    def trace(self) -> dict:
        cdef string i
        cdef dict out = {}
        cdef pair[string, vector[string]] it
        for it in self.ptr.container.trace: out[env(it.first)] = [env(i) for i in it.second]
        return out

    @property
    def extern_imports(self) -> dict:
        cdef string i
        cdef dict out = {}
        cdef pair[string, vector[string]] it
        for it in self.ptr.container.extern_imports: out[env(it.first)] = [env(i) for i in it.second]
        return out
