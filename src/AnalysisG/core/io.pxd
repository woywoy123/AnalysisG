# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.structs cimport data_t
from AnalysisG.core.meta cimport meta, Meta
from AnalysisG.core.notification cimport notification

cdef extern from "<io/io.h>" nogil:

    cdef cppclass io(notification):
        io() except+ nogil

        # hdf5 wrappers
        bool start(string filename, string read_write) except+ nogil
        void end() except+ nogil

        vector[string] dataset_names() except+ nogil
        bool has_dataset_name(string name) except+ nogil

        # ROOT wrappers
        map[string, long] root_size() except+ nogil
        void check_root_file_paths() except+ nogil
        bool scan_keys() except+ nogil
        void root_begin() except+ nogil
        void root_end() except+ nogil
        map[string, data_t*]* get_data() except+ nogil

        # ------ parameters ------- #
        string current_working_path
        string sow_name

        bool enable_pyami
        vector[string] trees
        vector[string] branches
        vector[string] leaves

        map[string, bool] root_files
        map[string, meta*] meta_data
        map[string, map[string, long]] tree_entries
        map[string, map[string, string]] leaf_typed
        map[string, map[string, map[string, vector[string]]]] keys
        string metacache_path


cdef class IO:
    cdef io* ptr
    cdef prg
    cdef dict meta_data
    cdef map[string, string] fnames
    cdef map[string, data_t*]* data_ops
    cdef map[string, bool] skip

    cpdef dict MetaData(self)
    cpdef void ScanKeys(self)
    cpdef void begin(self)
    cpdef void end(self)



