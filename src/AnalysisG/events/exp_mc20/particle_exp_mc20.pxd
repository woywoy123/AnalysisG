# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from AnalysisG.core.particle_template cimport *

cdef extern from "<exp_mc20/particles.h>":

    cdef cppclass top(particle_template):
        top() except+
        int barcode
        int status

    cdef cppclass child(particle_template):
        child() except+

        int barcode
        int status

    cdef cppclass physics_detector(particle_template):
        physics_detector() except+
        int parton_label
        int cone_label

        bool is_jet
        bool is_lepton
        bool is_photon

    cdef cppclass physics_truth(particle_template):
        physics_truth() except+

        int parton_label
        int cone_label

        bool is_jet
        bool is_lepton
        bool is_photon

    cdef cppclass electron(particle_template):
        electron() except+

        float d0
        int true_type
        float delta_z0
        int true_origin
        bool is_tight

    cdef cppclass muon(particle_template):
        muon() except+

        float d0
        int true_type
        float delta_z0
        int true_origin
        bool is_tight

    cdef cppclass jet(particle_template):
        jet() except+

        bool btag_65
        bool btag_70
        bool btag_77
        bool btag_85
        bool btag_90

        int flav
        int label

cdef class Top(ParticleTemplate):
    cdef top* p

    cdef public int barcode
    cdef public int status


cdef class Child(ParticleTemplate):
    cdef child* p

    cdef public int barcode
    cdef public int status

cdef class PhysicsDetector(ParticleTemplate):
    cdef physics_detector* p
    cdef public int parton_label
    cdef public int cone_label

    cdef public bool is_jet
    cdef public bool is_lepton
    cdef public bool is_photon

cdef class PhysicsTruth(ParticleTemplate):
    cdef physics_truth* p

    cdef public int parton_label
    cdef public int cone_label

    cdef public bool is_jet
    cdef public bool is_lepton
    cdef public bool is_photon

cdef class Electron(ParticleTemplate):
    cdef electron* p

    cdef public float d0
    cdef public int true_type
    cdef public float delta_z0
    cdef public int true_origin
    cdef public bool is_tight

cdef class Muon(ParticleTemplate):
    cdef muon* p

    cdef public float d0
    cdef public int true_type
    cdef public float delta_z0
    cdef public int true_origin
    cdef public bool is_tight

cdef class Jet(ParticleTemplate):
    cdef jet* p

    cdef public bool btag_65
    cdef public bool btag_70
    cdef public bool btag_77
    cdef public bool btag_85
    cdef public bool btag_90

    cdef public int flav
    cdef public int label


