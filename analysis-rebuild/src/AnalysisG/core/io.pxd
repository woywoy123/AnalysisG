# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.structs cimport particle_t
from AnalysisG.core.meta cimport meta, Meta


cdef extern from "<io/io.h>":
    enum data_enum:
        vvf "data_enum::vvf"
        vvl "data_enum::vvl"
        vvi "data_enum::vvi"
        vf  "data_enum::vf"
        vl  "data_enum::vl"
        vi  "data_enum::vi"
        vc  "data_enum::vc"
        f   "data_enum::f"
        l   "data_enum::l"
        i   "data_enum::i"
        ull "data_enum::ull"

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

        bool next(vector[vector[float]]* data) except +
        bool next(vector[vector[long ]]* data) except +
        bool next(vector[vector[int  ]]* data) except +

        bool next(vector[float]* data) except +
        bool next(vector[long]* data) except +
        bool next(vector[int]* data) except +
        bool next(vector[char ]* data) except +

        bool next(float* data) except +
        bool next(long* data) except +
        bool next(int* data) except +
        bool next(unsigned long long* data) except +

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

    cdef vector[float] vf
    if data.type == data_enum.vf and data.next(&vf): return {data.path : vf}

    cdef vector[long] vl
    if data.type == data_enum.vl and data.next(&vl): return {data.path : vl}

    cdef vector[int] vi
    if data.type == data_enum.vi and data.next(&vi): return {data.path : vi}

    cdef vector[char] vc
    if data.type == data_enum.vc and data.next(&vc): return {data.path : vc}

    cdef float f
    if data.type == data_enum.f and data.next(&f): return {data.path : f}

    cdef long l
    if data.type == data_enum.l and data.next(&l): return {data.path : l}

    cdef int i
    if data.type == data_enum.i and data.next(&i): return {data.path : i}

    cdef unsigned long long ull
    if data.type == data_enum.ull and data.next(&ull): return {data.path : ull}

    return {}

cdef class IO:
    cdef io* ptr
    cdef dict meta_data
    cdef map[string, data_t*]* data_ops

    cpdef dict MetaData(self)
    cpdef void ScanKeys(self)
    cpdef void begin(self)
    cpdef void end(self)



