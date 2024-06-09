# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from analysisg.core.structs cimport particle_t
from analysisg.core.meta cimport meta, Meta

cdef extern from "<io/io.h>":
    cdef enum class data_enum:
        vvf
        vvl
        vvi
        vf
        vl
        vi
        f
        l
        i

    cdef cppclass data_t:
        string leaf_name
        string branch_name
        string tree_name
        string leaf_type
        string path
        data_enum type
        int file_index
        long index

        void flush() except+

        bool next(float* data) except +
        bool next(vector[vector[float]]* data) except +
        bool next(vector[vector[long ]]* data) except +
        bool next(vector[vector[int  ]]* data) except +

    cdef cppclass io:
        io() except +

        # hdf5 wrappers
        bool start(string filename, string read_write) except +
        void end() except +

        vector[string] dataset_names() except +
        bool has_dataset_name(string name) except +

        void write(map[string, particle_t]* inpt, string set_name) except +

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


cdef inline dict switch_board(data_t* data):
    cdef vector[vector[float]] vvf
    if data.type == data_enum.vvf and data.next(&vvf): return {data.path : vvf}

    cdef vector[vector[long]] vvl
    if data.type == data_enum.vvl and data.next(&vvl): return {data.path : vvl}

    cdef vector[vector[int]] vvi
    if data.type == data_enum.vvi and data.next(&vvi): return {data.path : vvi}

    cdef float f
    if data.type == data_enum.f and data.next(&f): return {data.path : f}

    #if data.type == data_enum.vvf and data.next(&vvf): return {data.path : vvf}
    #if data.type == data_enum.vvf and data.next(&vvf): return {data.path : vvf}

    return {}

cdef class IO:
    cdef io* ptr
    cdef dict meta_data
    cdef map[string, data_t*]* data_ops

    cpdef dict MetaData(self)
    cpdef void ScanKeys(self)
    cpdef void begin(self)
    cpdef void end(self)



