# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.io cimport io
from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport Meta, meta
from AnalysisG.core.structs cimport data_t, data_enum, switch_board

from tqdm import tqdm
from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef class IO:
    def __cinit__(self):
        self.ptr = new io();
        self.data_ops = NULL
        self.meta_data = {}
        self.ptr.trigger_pcm()

    def __init__(self, root = []):
        self.prg = None
        if isinstance(root, list): self.Files = root
        elif isinstance(root, str): self.Files = [root]
        else: return

    def __dealloc__(self):
        self.ptr.root_end()
        del self.ptr

    def __len__(self):
        cdef pair[string, long] itx
        return max([itx.second for itx in self.ptr.root_size()] + [0])

    def __iter__(self):
        if self.prg is None: self.prg = tqdm(total = len(self), dynamic_ncols = True, leave = False)
        self.ptr.root_begin()
        self.data_ops = self.ptr.get_data()
        self.skip.clear()
        return self

    def __next__(self):
        cdef dict output = {}
        cdef pair[string, data_t*] itr
        cdef data_t* idx = NULL
        for itr in deref(self.data_ops):
            if self.skip[itr.first]: continue
            output |= switch_board(itr.second)
            self.skip[itr.first] = itr.second.next()
            if idx != NULL: continue
            idx = itr.second

        if not len(output):
            self.ptr.root_end()
            self.prg = None
            raise StopIteration

        cdef string di = deref(idx.fname)
        cdef string dx = self.fnames[di]
        if not dx.size():
            self.fnames[di] = enc(env(di).split("/")[-1])
            self.prg.set_description(env(self.fnames[di]))
            self.prg.refresh()
        self.prg.update(1)
        output["filename"] = di
        return output

    cpdef dict MetaData(self):
        self.meta_data = {}

        cdef Meta data
        cdef pair[string, meta*] itr
        for itr in self.ptr.meta_data:
            data = Meta()
            data.ptr.metacache_path = itr.second.metacache_path
            if self.EnablePyAMI: data.__meta__(itr.second)
            self.meta_data[env(itr.first)] = data
        return self.meta_data

    @property
    def Verbose(self): return not self.ptr.shush
    @Verbose.setter
    def Verbose(self, bool val): self.ptr.shush = not val

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

    @property
    def SumOfWeightsTreeName(self): return env(self.ptr.sow_name)
    @SumOfWeightsTreeName.setter
    def SumOfWeightsTreeName(self, str val): self.ptr.sow_name = enc(val)

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
