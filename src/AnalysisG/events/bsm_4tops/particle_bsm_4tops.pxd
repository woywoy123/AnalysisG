# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.particle_template cimport particle_template

cdef extern from "<bsm_4tops/particles.h>":

    cdef cppclass top(particle_template):
        top() except+
        bool from_res
        int status
        vector[truthjet*] TruthJets
        vector[jet*] Jets

    cdef cppclass top_children(particle_template):
        top_children() except+
        int top_index
        bool from_res

    cdef cppclass truthjet(particle_template):
        truthjet() except+
        int top_quark_count
        int w_boson_count
        bool from_res

        vector[int] top_index

        vector[top*] Tops
        vector[truthjetparton*] Parton


    cdef cppclass truthjetparton(particle_template):
        truthjetparton() except+

        int truthjet_index
        vector[int] topchild_index

    cdef cppclass jet(particle_template):
        jet() except+
        vector[top*] Tops
        vector[jetparton*] Parton

        vector[int] top_index
        bool btag_DL1r_60
        bool btag_DL1_60
        bool btag_DL1r_70
        bool btag_DL1_70
        bool btag_DL1r_77
        bool btag_DL1_77
        bool btag_DL1r_85
        bool btag_DL1_85

        float DL1_b
        float DL1_c
        float DL1_u
        float DL1r_b
        float DL1r_c
        float DL1r_u

    cdef cppclass jetparton(particle_template):
        jetparton() except+

        int jet_index
        vector[int] topchild_index

    cdef cppclass electron(particle_template):
        electron() except+

    cdef cppclass muon(particle_template):
        muon() except+

