# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string

from AnalysisG.core.structs cimport *
from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.particle_template cimport particle_template

cdef extern from "<templates/selection_template.h>" nogil:
    cdef cppclass selection_template:
        selection_template() except+ nogil

        string name
        string hash
        string tree
        double index

        selection_template* build(event_template*) except+ nogil
        bool selection(event_template*) except+ nogil
        bool strategy(event_template*) except+ nogil
        void merge(selection_template*) except+ nogil

        void sum(vector[particle_template*]* ch, particle_template** out) except+ nogil
        particle_template* sum(map[string, particle_template*]* ch) except+ nogil

        vector[map[string, float]] reverse_hash(vector[string]* hashes) except+ nogil
        bool CompileEvent() except+ nogil

        bool operator == (selection_template& p) except+ nogil
        map[string, map[string, float]] passed_weights
        map[string, meta_t] matched_meta

        void switch_board(particle_enum attrs, particle_template* ptr, vector[vector[double]]* data); 
        void switch_board(particle_enum attrs, particle_template* ptr, vector[int]*    data); 
        void switch_board(particle_enum attrs, particle_template* ptr, vector[double]* data); 
        void switch_board(particle_enum attrs, particle_template* ptr, vector[bool]*   data); 

cdef class SelectionTemplate:
    cdef selection_template* ptr
    cdef void transform_dict_keys(self)
    cdef public dict root_leaves
