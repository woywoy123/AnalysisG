# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string

cdef class SelectionTemplate:
    def __cinit__(self):
        if type(self) is not SelectionTemplate: return
        self.ptr = new selection_template()

    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt[i])
            except KeyError: continue

    def __dealloc__(self):
        if type(self) is not SelectionTemplate: return
        del self.ptr

    def __hash__(self):
        return int(string(self.ptr.hash).substr(0, 8), 0)

    def __reduce__(self):
        cdef list keys = self.__dir__()
        cdef dict out = {i : getattr(self, i) for i in keys if not i.startswith("__")}
        return self.__class__, (out,)

    cdef void transform_dict_keys(self): pass
