# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.meta cimport meta, Meta
from AnalysisG.core.structs cimport data_t

cdef extern from "<io/io.h>":

    cdef cppclass io:
        io() except +

        # hdf5 wrappers
        bool start(string filename, string read_write) except +
        void end() except +

        vector[string] dataset_names() except +
        bool has_dataset_name(string name) except +

        # ROOT wrappers
        map[string, long] root_size() except +
        void check_root_file_paths() except +
        bool scan_keys() except +
        void root_begin() except +
        void root_end() except +
        map[string, data_t*]* get_data() except +

        # ------ parameters ------- #
        string current_working_path

        bool enable_pyami
        string metacache_path
        vector[string] trees
        vector[string] branches
        vector[string] leaves

        map[string, bool] root_files
        map[string, meta*] meta_data
        map[string, map[string, long]] tree_entries
        map[string, map[string, string]] leaf_typed
        map[string, map[string, map[string, vector[string]]]] keys


cdef class IO:
    cdef io* ptr
    cdef dict meta_data
    cdef map[string, data_t*]* data_ops
    cdef map[string, bool] skip

    cpdef dict MetaData(self)
    cpdef void ScanKeys(self)
    cpdef void begin(self)
    cpdef void end(self)



