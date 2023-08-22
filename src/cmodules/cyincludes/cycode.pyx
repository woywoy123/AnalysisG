# distuils: language = c++
# cython: language_level = 3
from libcpp.string cimport string
from cycode cimport CyCode, ExportCode
from libcpp cimport bool
from libcpp cimport map

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
        return self.ptr==o.ptr

    def __getstate__(self) -> ExportCode:
        return self.ptr.MakeMapping()

    def __setstate__(self, inpt):
        pass

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

        self.ptr.is_class = self.is_class
        self.ptr.is_function = self.is_class
        self.ptr.is_callable = self.is_callable
        self.ptr.is_initialized = self.is_initialized
        self.ptr.function_name = enc(self.function_name)
        self.ptr.class_name = enc(self.class_name)
        self.ptr.source_code = enc(self.source_code)
        self.ptr.object_code = enc(self.object_code)

        self.ptr.co_vars = [enc(i) for i in self.co_vars]
        self.ptr.input_params = [enc(i) for i in self.input_params]

    def __getclass__(self):
        try: self._tmp = self._x()
        except TypeError: pass
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
    def hash(self) -> str:
        self.ptr.Hash()
        return env(self.ptr.hash)

    def add_dependency(self, ExportCode inpt):
        self.ptr.dependency[inpt.hash] = inpt
