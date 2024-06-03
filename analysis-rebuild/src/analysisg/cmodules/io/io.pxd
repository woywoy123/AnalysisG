# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from analysisg.cmodules.structs.structs cimport particle_t

cdef extern from "<io/io.h>":

    cdef cppclass io:
        io() except +

        # hdf5 wrappers
        bool start(string filename, string read_write) except +
        void end() except +

        vector[string] dataset_names() except +
        bool has_dataset_name(string name) except +

        void write(map[string, particle_t]* inpt, string set_name) except +

        # ROOT wrappers
        void check_root_file_paths() except +
        void scan_keys() except +

        # ------ parameters ------- #
        string current_working_path

        bool enable_pyami
        string metacache_path
        vector[string] trees
        vector[string] branches
        vector[string] leaves
        map[string, bool] root_files

cdef class IO:
    cdef io* ptr
    cpdef void scan_keys(self)



