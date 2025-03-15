# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "matching.h":
    cdef cppclass particle(particle_template):
        particle() except+
        string root_hash

    struct packet_t:
        vector[particle*] truth_tops
        vector[particle*] children_tops
        vector[particle*] truth_jets
        vector[particle*] jets_children
        vector[particle*] jets_leptons

    cdef cppclass matching(selection_template):
        matching() except +
        vector[packet_t] output

cdef class TopMatching(SelectionTemplate):
    cdef matching* tt

    cdef list make_particle(self, vector[particle*] px)

    cdef public list truth_tops
    cdef public list top_children
    cdef public list truth_jets
    cdef public list jets_children
    cdef public list jets_leptons
