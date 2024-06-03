# cython: language_level = 3

from .io cimport io
from analysisg.cmodules.tools.tools cimport *

from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef class IO:
    def __cinit__(self): self.ptr = new io();
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    @property
    def EnablePyAMI(self): return self.ptr.enable_pyami

    @EnablePyAMI.setter
    def EnablePyAMI(self, val): self.ptr.enable_pyami = val

    @property
    def MetaCachePath(self): return env(self.ptr.metacache_path)

    @MetaCachePath.setter
    def MetaCachePath(self, val): self.ptr.metacache_path = enc(val)

    @property
    def Trees(self): return env_vec(&self.ptr.trees)

    @Trees.setter
    def Trees(self, val):
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.trees = enc_list(val)

    @property
    def Branches(self): return env_vec(&self.ptr.branches)

    @Branches.setter
    def Branches(self, val):
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.branches = enc_list(val)

    @property
    def Leaves(self): return env_vec(&self.ptr.leaves)

    @Leaves.setter
    def Leaves(self, val):
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.leaves = enc_list(val)

    @property
    def Files(self):
        cdef pair[string, bool] itr
        cdef list output = []
        self.ptr.check_root_file_paths()
        for itr in self.ptr.root_files:
            if not itr.second: continue
            output.append(env(itr.first))
        return output

    @Files.setter
    def Files(self, val):
        cdef list f_val = []
        if isinstance(val, str): f_val.append(val)
        elif isinstance(val, list): f_val += val
        else: return
        if not len(f_val): self.ptr.root_files.clear()

        for i in f_val:
            if not isinstance(i, str): continue
            self.ptr.root_files[enc(i)] = False
        self.ptr.check_root_file_paths()

    cpdef void scan_keys(self): self.ptr.scan_keys()

