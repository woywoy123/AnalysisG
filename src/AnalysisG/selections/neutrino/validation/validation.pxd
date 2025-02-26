# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "validation.h":
    cdef cppclass nu(particle_template):
        nu() except +
        double distance

    cdef cppclass tquark(particle_template):
        topquark() except +

    cdef cppclass bquark(particle_template):
        bquark() except +

    cdef cppclass lepton(particle_template):
        lepton() except +

    cdef cppclass boson(particle_template):
        boson() except +

    struct package:
        double met
        double phi

        vector[nu*]     truth_nus
        vector[tquark*] truth_tops
        vector[boson*]  truth_bosons

        vector[lepton*] reco_leptons
        vector[boson*]  reco_bosons

        vector[lepton*] truth_leptons
        vector[bquark*] truth_bquarks

        vector[bquark*] truth_bjets
        vector[tquark*] truth_jets_top

        vector[bquark*] bjets
        vector[tquark*] jets_top
        vector[tquark*] lepton_jets_top

        vector[nu*] c1_reconstructed_children_nu
        vector[nu*] c1_reconstructed_truthjet_nu
        vector[nu*] c1_reconstructed_jetchild_nu
        vector[nu*] c1_reconstructed_jetlep_nu

        vector[nu*] c2_reconstructed_children_nu
        vector[nu*] c2_reconstructed_truthjet_nu
        vector[nu*] c2_reconstructed_jetchild_nu
        vector[nu*] c2_reconstructed_jetlep_nu

    cdef cppclass validation(selection_template):
        validation() except +
        map[string, package] data_out

cdef class Validation(SelectionTemplate):
    cdef validation* tt
    cdef list build_p0(self, vector[tquark*]* inpt)
    cdef list build_p1(self, vector[bquark*]* inpt)
    cdef list build_p2(self, vector[lepton*]* inpt)
    cdef list build_p3(self, vector[boson*]*  inpt)
    cdef list build_p4(self, vector[nu*]*     inpt)

    cdef public list met
    cdef public list phi

    cdef public list truth_nus
    cdef public list truth_tops
    cdef public list truth_bosons

    cdef public list reco_leptons
    cdef public list reco_bosons

    cdef public list truth_leptons
    cdef public list truth_bquarks

    cdef public list truth_bjets
    cdef public list truth_jets_top

    cdef public list bjets
    cdef public list jets_top
    cdef public list lepton_jets_top

    cdef public list c1_reconstructed_children_nu
    cdef public list c1_reconstructed_truthjet_nu
    cdef public list c1_reconstructed_jetchild_nu
    cdef public list c1_reconstructed_jetlep_nu

    cdef public list c2_reconstructed_children_nu
    cdef public list c2_reconstructed_truthjet_nu
    cdef public list c2_reconstructed_jetchild_nu
    cdef public list c2_reconstructed_jetlep_nu


