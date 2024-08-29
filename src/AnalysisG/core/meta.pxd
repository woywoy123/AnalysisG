# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.structs cimport meta_t
from AnalysisG.core.notification cimport *

cdef extern from "<meta/meta.h>":
    cdef cppclass meta:
        meta() except +
        meta_t meta_data
        string metacache_path

cdef class Meta:
    cdef meta* ptr
    cdef __meta__(self, meta* met)

cdef class ami_client:
    cdef client
    cdef file_cache
    cdef notification* nf

    cdef str type_
    cdef list dsids
    cdef dict datas
    cdef dict infos

    cdef bool loadcache(self, Meta obj)
    cdef void savecache(self, Meta obj)
    cdef void dressmeta(self, Meta obj, str dset_name)
    cdef void list_datasets(self, Meta obj)



