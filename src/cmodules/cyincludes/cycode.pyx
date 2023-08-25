# distuils: language = c++
# cython: language_level = 3
from libcpp.string cimport string
from libcpp cimport bool
from libcpp cimport map

from cycode cimport CyCode
from cytypes cimport code_t

import inspect
import os, sys

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Code:

    cdef CyCode* ptr
    cdef public bool is_class
    cdef public bool is_function
    cdef public bool is_callable
    cdef public bool is_initialized

    cdef public str function_name
    cdef public str class_name

    cdef public list defaults
    cdef public list co_vars
    cdef public dict input_params
    cdef public str source_code
    cdef public str object_code
    cdef public _x

    def __cinit__(self):
        self.ptr = new CyCode()
        self.is_class = False
        self.is_function = False
        self.is_callable = False
        self.is_initialized = False
        self.function_name = ""
        self.class_name = ""
        self.source_code = ""
        self.object_code = ""

        self.defaults = []
        self.co_vars = []
        self.input_params = {}

    def __init__(self, instance = None):
        if instance is None: return
        self._x = instance
        self.__getbasic__()

    def __dealloc__(self):
        del self.ptr

    def __hash__(self) -> int:
        return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef Code o = other
        return self.ptr[0]==o.ptr[0]

    def __getstate__(self) -> code_t:
        return self.ptr.ExportCode()

    def __setstate__(self, code_t inpt):
        self.ptr.ImportCode(inpt)
        self.is_class           = self.ptr.container.is_class
        self.is_function        = self.ptr.container.is_function
        self.is_callable        = self.ptr.container.is_callable
        self.is_initialized     = self.ptr.container.is_initialized
        self.function_name      = env(self.ptr.container.function_name)
        self.class_name         = env(self.ptr.container.class_name)
        self.source_code        = env(self.ptr.container.source_code)
        self.object_code        = env(self.ptr.container.object_code)
        cdef string key, var
        cdef list keys = self.ptr.container.co_vars
        cdef list vals = self.ptr.container.input_params
        for key, var in zip(keys, vals):
            self.input_params[env(key)] = env(var)
            self.co_vars.append(env(key))

    def is_self(self, inpt) -> bool:
        return isinstance(inpt, Code)

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

        try: self.object_code = inspect.getsource(self._x)
        except TypeError: 
            self.object_code = inspect.getsource(self._x.__class__)

        self.ptr.container.is_class = self.is_class
        self.ptr.container.is_function = self.is_class
        self.ptr.container.is_callable = self.is_callable
        self.ptr.container.is_initialized = self.is_initialized
        self.ptr.container.function_name = enc(self.function_name)
        self.ptr.container.class_name = enc(self.class_name)
        self.ptr.container.source_code = enc(self.source_code)
        self.ptr.container.object_code = enc(self.object_code)

        self.ptr.container.co_vars = [enc(i) for i in self.co_vars]
        self.ptr.container.input_params = [enc(i) for i in self.input_params]
        self.ptr.Hash()

    def __getclass__(self):
        ac = type(self._x.__init__).__name__
        if ac == "function": self.is_initialized = False
        else: self.is_initialized = True
        self.__getinputs__(self._x.__init__)

    def __getfunction__(self):
        self.__getinputs__(self._x)

    def __getinputs__(self, val):
        cdef int start = 0
        if self.is_class: start = 1
        self.co_vars  = list(val.__code__.co_varnames)[start:]
        if val.__defaults__ is not None:
            self.defaults = list(val.__defaults__)
            self.input_params = {key : None for key in self.co_vars}
        else: return
        self.defaults.reverse()
        self.co_vars.reverse()

        for key, val in zip(self.co_vars, self.defaults):
            self.input_params[key] = val

    @property
    def InstantiateObject(self):
        exec(env(self.ptr.source_code))
        if self.ptr.is_class: return locals()[env(self.ptr.class_name)]()
        else: return locals()[env(self.ptr.function_name)]


    @property
    def hash(self) -> str:
        self.ptr.Hash()
        return env(self.ptr.hash)
