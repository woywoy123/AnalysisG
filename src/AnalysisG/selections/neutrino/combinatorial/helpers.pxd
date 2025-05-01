# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport int, bool

cdef cppclass neutrino(particle_template):
    double ellipse 
    double chi2

    int matched_bquark
    int matched_lepton

cdef struct sets:
    particle_template* tru_b  
    particle_template* tru_l  
    particle_template* tru_nu 
    neutrino*          rec_nu 

    particle_template* tru_top
    particle_template* rec_top

    particle_template* tru_wboson
    particle_template* rec_wboson
    
    double  ellipse 
    double  chi2
    string  symbolic

cdef struct atomic:
    int n_correct    # <- correct matched
    int n_b_swapped  # <- only bs are swapped
    int n_l_swapped  # <- only leptons are swapped
    int n_bl_swapped # <- both bs and leptons are swapped
    int n_unmatched  # <- Unmatched or fake neutrino
    int n_non_nunu   # <- triggered on a non-dilepton event

    int n_tru_nu  
    int n_rec_nu  

    int num_sols  
    int merged_jet

    map[int, sets] correct   
    map[int, sets] swapped_bs
    map[int, sets] swapped_bl
    map[int, sets] swapped_ls
    map[int, sets] fake_nus  
    map[int, sets] loss
    string symbolics
    bool ignore


cdef struct export_t:
   map[string, vector[int]] n_correct   
   map[string, vector[int]] n_b_swapped 
   map[string, vector[int]] n_l_swapped 
   map[string, vector[int]] n_bl_swapped
   map[string, vector[int]] n_unmatched 
   map[string, vector[int]] n_non_nunu  

   map[string, vector[int]] n_tru_nu 
   map[string, vector[int]] n_rec_nu 
   map[string, vector[int]] num_sols 
   map[string, vector[int]] merged_jet
  
   map[int, map[string, vector[sets]]] loss
   map[int, map[string, vector[sets]]] correct       
   map[int, map[string, vector[sets]]] swapped_bs 
   map[int, map[string, vector[sets]]] swapped_bl 
   map[int, map[string, vector[sets]]] swapped_ls 
   map[int, map[string, vector[sets]]] fake_nus   

cdef cppclass event_t:
    map[string, map[int, vector[particle_template*]]] truth_tops
    map[string, map[int, vector[particle_template*]]] reco_tops

cdef struct container:
    map[string, vector[atomic]] atomics
    map[string, vector[atomic]] chi2_atomics
    selection_template* slx

cdef void assign_particles(string name, vector[particle_template*]* pte, event_t* ev)
cdef void assign_neutrinos(string name, vector[neutrino*]* pt1, vector[neutrino*]* pt2, event_t* ev)

cdef void add_container(string name, container* con, event_t* ev, bool ch2)
cdef export_t get_export(string name, container* con, bool ch2)


cdef vector[particle_template*] make_particle(vector[vector[double]]* pmu, vector[int]* pdgid)
cdef vector[neutrino*] make_neutrino(vector[vector[double]]* pmu, vector[int]* lep, vector[int]* bq, vector[double]* elp, vector[double]* chi)

