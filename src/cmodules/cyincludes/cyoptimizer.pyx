# distutils: language = c++
# cython: language_level = 3
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

from cytools cimport env, enc, map_to_dict
from cyoptimizer cimport CyOptimizer
from cytypes cimport folds_t

from torch_geometric.data import Batch
import psutil
import torch
import h5py

cdef class cOptimizer:
    cdef CyOptimizer* ptr
    cdef dict _kModel
    cdef dict _kOptim
    cdef dict _cached

    def __cinit__(self):
        self.ptr = new CyOptimizer()
        self._cached = {}

    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    def length(self):
        return map_to_dict(<map[string, int]>self.ptr.fold_map())

    def GetHDF5Hashes(self, str path) -> bool:
        if path.endswith(".hdf5"): pass
        else: path += ".hdf5"

        try: f = h5py.File(path, "r")
        except FileNotFoundError: return False

        cdef str hash_, key_
        cdef folds_t fold_hash

        for hash_ in f:
            fold_hash = folds_t()
            fold_hash.event_hash = enc(hash_)
            try:
                fold_hash.test = f[hash_].attrs["test"]
                self.ptr.register_fold(&fold_hash)
                continue
            except KeyError: fold_hash.test = False

            try: fold_hash.train = f[hash_].attrs["train"]
            except KeyError: fold_hash.train = False
            for key_ in f[hash_].attrs:
                if not key_.startswith("k-"): continue
                fold_hash.kfold = int(key_[2:])
                fold_hash.train = f[hash_].attrs[key_]
                if fold_hash.train: pass
                else: fold_hash.evaluation = True
                self.ptr.register_fold(&fold_hash)

    def UseAllHashes(self, dict inpt):
        cdef str key, hash_
        cdef dict data = inpt["graph"]
        cdef folds_t fold_hash
        for key in data:
            for hash_ in data[key]:
                fold_hash = folds_t()
                fold_hash.kfold = 1
                fold_hash.train = True
                fold_hash.event_hash = enc(hash_)
                self.ptr.register_fold(&fold_hash)
            data[key] = None

    def FetchTraining(self, int kfold, int batch_size):
        return self.ptr.fetch_train(kfold, batch_size)

    def MakeBatch(self, sampletracer, vector[string] batch, int kfold, int index, int max_percent = 80):
        cdef str k, j
        cdef string k_
        cdef list lsts
        cdef tuple cuda
        cdef float gpu = -1
        cdef float ram = -1

        ram = psutil.virtual_memory().percent
        if ram > max_percent:
            lsts = [i.hash for i in sampletracer.makelist()]
            sampletracer.FlushGraphs(lsts)
            self.ptr.flush_train([enc(k) for k in lsts], kfold)

        if sampletracer.Device != "cpu":
            cuda = torch.cuda.mem_get_into()
            gpu = (1 - cuda[0]/cuda[1])*100

        if gpu > max_percent or ram > max_percent:
            lsts = [list(self._cached[k].values()) for k in self._cached]
            for k in sum(lsts, []): del k
            self._cached = {}
            if sampletracer.Device == "cpu": pass
            torch.cuda.empty_cache()

        lsts = [env(k_) for k_ in self.ptr.check_train(batch, kfold)]
        if len(lsts): sampletracer.RestoreGraphs(lsts)

        if kfold not in self._cached: self._cached[kfold] = {}
        if index not in self._cached[kfold]: self._cached[kfold][index] = None
        if self._cached[kfold][index] is not None: return self._cached[kfold][index]

        lsts = [env(k_) for k_ in batch]
        if len(lsts) == 1: lsts = [sampletracer[lsts]]
        else: lsts = sampletracer[lsts]
        lsts = [i.release_graph().to(sampletracer.Device) for i in lsts]
        self._cached[kfold][index] = Batch().from_data_list(lsts)
        return self._cached[kfold][index]

    def UseTheseFolds(self, list inpt):
        self.ptr.use_folds = <vector[int]>inpt

    @property
    def kFolds(self): return self.ptr.use_folds


