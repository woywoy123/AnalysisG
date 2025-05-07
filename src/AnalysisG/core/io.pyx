/**
 * @file io.pyx
 * @brief Provides input/output operations for the AnalysisG framework.
 */

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

/**
 * @class IO
 * @brief Handles file input/output operations and metadata management.
 */
cdef class IO:
    def __cinit__(self):
        self.ptr = new io();
        self.data_ops = NULL
        self.meta_data = {}
        self.ptr.trigger_pcm()

    def __init__(self, root = []):
        """
        @brief Initializes the IO class with a list of root files or a single root file.
        @param root A list of root files or a single root file.
        """
        self.prg = None
        if isinstance(root, list): self.Files = root
        elif isinstance(root, str): self.Files = [root]
        else: return

    def __dealloc__(self):
        """
        @brief Deallocates resources used by the IO class.
        """
        self.ptr.root_end()
        del self.ptr

    def __len__(self):
        """
        @brief Returns the maximum size of the root files.
        @return The maximum size of the root files.
        """
        cdef pair[string, long] itx
        return max([itx.second for itx in self.ptr.root_size()] + [0])

    def __iter__(self):
        """
        @brief Initializes the iterator for the IO class.
        @return The IO class instance as an iterator.
        """
        if self.prg is None: self.prg = tqdm(total = len(self), dynamic_ncols = True, leave = False)
        self.ptr.root_begin()
        self.data_ops = self.ptr.get_data()
        self.skip.clear()
        return self

    def __next__(self):
        """
        @brief Retrieves the next item in the iteration.
        @return A dictionary containing the next data item.
        @throws StopIteration When the iteration is complete.
        """
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
        """
        @brief Retrieves metadata associated with the IO class.
        @return A dictionary containing metadata.
        """
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
    def Verbose(self):
        """
        @brief Gets the verbosity state.
        @return True if verbose, False otherwise.
        """
        return not self.ptr.shush

    @Verbose.setter
    def Verbose(self, bool val):
        """
        @brief Sets the verbosity state.
        @param val True to enable verbose, False to disable.
        """
        self.ptr.shush = not val

    @property
    def EnablePyAMI(self):
        """
        @brief Gets the state of PyAMI.
        @return True if PyAMI is enabled, False otherwise.
        """
        return self.ptr.enable_pyami

    @EnablePyAMI.setter
    def EnablePyAMI(self, val):
        """
        @brief Sets the state of PyAMI.
        @param val True to enable PyAMI, False to disable.
        """
        self.ptr.enable_pyami = val

    @property
    def MetaCachePath(self):
        """
        @brief Gets the path to the metadata cache.
        @return The metadata cache path as a string.
        """
        return env(self.ptr.metacache_path)

    @MetaCachePath.setter
    def MetaCachePath(self, val):
        """
        @brief Sets the path to the metadata cache.
        @param val The metadata cache path as a string.
        """
        self.ptr.metacache_path = enc(val)

    @property
    def Trees(self):
        """
        @brief Gets the list of trees.
        @return A list of tree names.
        """
        return env_vec(&self.ptr.trees)

    @property
    def SumOfWeightsTreeName(self):
        """
        @brief Gets the name of the sum of weights tree.
        @return The name of the sum of weights tree as a string.
        """
        return env(self.ptr.sow_name)

    @SumOfWeightsTreeName.setter
    def SumOfWeightsTreeName(self, str val):
        """
        @brief Sets the name of the sum of weights tree.
        @param val The name of the sum of weights tree as a string.
        """
        self.ptr.sow_name = enc(val)

    @Trees.setter
    def Trees(self, val):
        """
        @brief Sets the list of trees.
        @param val A list of tree names or a single tree name.
        """
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.trees = enc_list(val)

    @property
    def Branches(self):
        """
        @brief Gets the list of branches.
        @return A list of branch names.
        """
        return env_vec(&self.ptr.branches)

    @Branches.setter
    def Branches(self, val):
        """
        @brief Sets the list of branches.
        @param val A list of branch names or a single branch name.
        """
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.branches = enc_list(val)

    @property
    def Leaves(self):
        """
        @brief Gets the list of leaves.
        @return A list of leaf names.
        """
        return env_vec(&self.ptr.leaves)

    @Leaves.setter
    def Leaves(self, val):
        """
        @brief Sets the list of leaves.
        @param val A list of leaf names or a single leaf name.
        """
        if isinstance(val, str): val = [val]
        elif isinstance(val, list): pass
        else: return
        self.ptr.leaves = enc_list(val)

    @property
    def Files(self):
        """
        @brief Gets the list of root files.
        @return A list of root file paths.
        """
        cdef pair[string, bool] itr
        cdef list output = []
        self.ptr.check_root_file_paths()
        for itr in self.ptr.root_files:
            if not itr.second: continue
            output.append(env(itr.first))
        return output

    @Files.setter
    def Files(self, val):
        """
        @brief Sets the list of root files.
        @param val A list of root file paths or a single root file path.
        """
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
        """
        @brief Gets the keys from the root files.
        @return A dictionary containing the keys.
        """
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

    cpdef void ScanKeys(self):
        """
        @brief Scans the keys in the root files.
        """
        self.ptr.scan_keys()

    cpdef void begin(self):
        """
        @brief Begins the I/O operation by initializing the root structure.
        """
        self.ptr.root_begin()

    cpdef void end(self):
        """
        @brief Ends the I/O operation by finalizing the root structure.
        """
        self.ptr.root_end()
