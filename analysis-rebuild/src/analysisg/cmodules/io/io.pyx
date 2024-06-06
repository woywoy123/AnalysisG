# cython: language_level = 3

from .io cimport io, data_t, data_enum, switch_board
from analysisg.cmodules.tools.tools cimport *

from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef class IO:
    def __cinit__(self): self.ptr = new io();
    def __init__(self, list root = []):
        if len(root): self.Files = root
        else: return

    def __dealloc__(self): del self.ptr
    def __len__(self):
        cdef pair[string, long] itx
        return max([itx.second for itx in self.ptr.root_size()])

    def __iter__(self):
        self.ptr.root_begin()
        self.data_ops = self.ptr.get_data()
        return self

    def __next__(self):
        cdef dict output = {}
        cdef pair[string, data_t*] itr
        for itr in deref(self.data_ops): output |= switch_board(itr.second)
        if not len(output): raise StopIteration
        return output

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

    @property
    def Keys(self):
        self.ptr.scan_keys()
        cdef dict output = {}

        cdef str fname
        cdef str fname_
        cdef str fname__

        cdef pair[string, map[string, map[string, vector[string]]]] ite
        cdef pair[string, map[string, vector[string]]] ite_
        cdef pair[string, vector[string]] ite__

        for ite in self.ptr.keys:
            fname = env(ite.first)
            output[fname] = {}
            for ite_ in ite.second:
                fname_ = env(ite_.first)
                if fname_ not in output[fname]: output[fname][fname_] = {}

                for ite__ in ite_.second:
                    fname__ = env(ite__.first)
                    output[fname][fname_][fname__] = env_vec(&ite__.second)
        return output

    cpdef void ScanKeys(self): self.ptr.scan_keys()
    cpdef void begin(self): self.ptr.root_begin()
    cpdef void end(self): self.ptr.root_end()
