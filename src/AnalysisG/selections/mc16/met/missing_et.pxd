# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "met.h":
    cdef cppclass met(selection_template):
        met() except +
        double masstop
        double massw
        double perturb
        double distance
        int    steps

cdef struct container_t:
    double missing_evn_px 
    double missing_evn_py 
    double missing_det_px 
    double missing_det_py 
    double missing_det_pz 
    double missing_nus_px 
    double missing_nus_py 
    double missing_nus_pz 
    double num_neutrino    
    double num_leptons       
    double num_leptons_reco 

    vector[int]       top_index  
    vector[double] mass_tru_top  
    vector[double] mass_tru_top3 

    vector[vector[double]] tru_nu  

    vector[double] chi2_sols 
    vector[int] top_index_sols1 
    vector[int] top_index_sols2 

    vector[double] top_mass_sols1 
    vector[double] top_mass_sols2 

    vector[vector[double]] nu1 
    vector[vector[double]] nu2 

    vector[vector[double]] agnR_pln_w1
    vector[vector[double]] agnR_pln_w2 
    vector[vector[double]] agnR_pln_t1 
    vector[vector[double]] agnR_pln_t2 
    vector[vector[double]] agnR_pln_t1xt2 
    vector[double]         agnR_dR_nu_lep1 
    vector[double]         agnR_dR_nu_lep2 
    vector[bool]           agnR_has_null

    vector[vector[double]] agnT_pln_w1
    vector[vector[double]] agnT_pln_w2 
    vector[vector[double]] agnT_pln_t1 
    vector[vector[double]] agnT_pln_t2 
    vector[vector[double]] agnT_pln_t1xt2 
    vector[double]         agnT_dR_nu_lep1 
    vector[double]         agnT_dR_nu_lep2 
    vector[bool]           agnT_has_null



cdef class MissingET(SelectionTemplate):
    cdef met* tt

    cdef public vector[container_t] data
    cdef map[string, int] indx

