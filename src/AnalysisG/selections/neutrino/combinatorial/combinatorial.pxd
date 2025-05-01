# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *
#from cython.operator cimport dereference as dref
#from .helpers cimport *

cdef extern from "combinatorial.h":
    cdef cppclass combinatorial(selection_template):
        combinatorial() except +
        int num_device
        double masstop
        double massw

cdef cppclass neutrino(particle_template):
    double ellipse 
    double chi2

    int matched_bquark
    int matched_lepton
     

cdef class Neutrino(ParticleTemplate):
    cdef neutrino* nux

cdef class Particle(ParticleTemplate): 
    pass


cdef class NuNuCombinatorial(SelectionTemplate):
    cdef combinatorial* tt

    cdef int ix
    cdef public int lx

    cdef map[string, vector[vector[vector[double]]]] pmu
    cdef map[string, vector[vector[int]]] pdgid

    cdef map[string, vector[vector[int]]] matched_bq
    cdef map[string, vector[vector[int]]] matched_lp
    
    cdef map[string, vector[vector[double]]] ellipse 
    cdef map[string, vector[vector[double]]] chi2_nu1
    cdef map[string, vector[vector[double]]] chi2_nu2
    cdef map[string, vector[vector[vector[double]]]] pmu_nu1
    cdef map[string, vector[vector[vector[double]]]] pmu_nu2


cdef inline particle_template* make_particle(vector[double]* pmu, int pdgid):
    cdef particle_template* ptx = new particle_template()
    ptx.pt  = pmu.at(0); ptx.eta = pmu.at(1)
    ptx.phi = pmu.at(2); ptx.e   = pmu.at(3)
    ptx.pdgid = pdgid
    return ptx

cdef inline neutrino* make_neutrino(vector[double]* pmu, int lep, int bq, double elp, double chi):
    cdef neutrino* ptx = new neutrino()
    ptx.pt  = pmu.at(0); ptx.eta = pmu.at(1)
    ptx.phi = pmu.at(2); ptx.e   = pmu.at(3)
    ptx.matched_bquark = bq; ptx.matched_lepton = lep
    ptx.ellipse = elp; ptx.chi2 = chi; 
    return ptx

cdef inline void loader(NuNuCombinatorial vl, tuple data):
    cdef str name = data[0]
    cdef string name_
    if   "top_children" in name: name_ = b"top_children"
    elif "truthjet"     in name: name_ = b"truthjet"    
    elif "jetchildren"  in name: name_ = b"jetchildren" 
    elif "jetleptons"   in name: name_ = b"jetleptons"  
    else: print("invalid")

    if "matched_bquark" in name: vl.matched_bq[name_].push_back(<vector[int]>(data[1]));  return
    if "matched_lepton" in name: vl.matched_lp[name_].push_back(<vector[int]>(data[1]));  return
    if "distance"       in name: vl.ellipse[name_].push_back(<vector[double]>(data[1]));  return
    if "pdgid"          in name: vl.pdgid[name_].push_back(<vector[int]>(data[1]));       return
    if "nu1_chi2"       in name: vl.chi2_nu1[name_].push_back(<vector[double]>(data[1])); return
    if "nu2_chi2"       in name: vl.chi2_nu2[name_].push_back(<vector[double]>(data[1])); return
    if "nu1_pmu"        in name: vl.pmu_nu1[name_].push_back(<vector[vector[double]]>(data[1])); return
    if "nu2_pmu"        in name: vl.pmu_nu2[name_].push_back(<vector[vector[double]]>(data[1])); return
    if "pmu"            in name: vl.pmu[name_].push_back(<vector[vector[double]]>(data[1]));     return


#cdef inline void delete(map[int, vector[particle_template*]]* px):
#    cdef int x 
#    cdef pair[int, vector[particle_template*]] itx
#    for itx in dref(px):
#        for x in range(itx.second.size()): del itx.second[x]
#

#cdef class NuNuCombinatorial(SelectionTemplate):
#    cdef combinatorial* tt
#    cdef container con
#   
#    cdef map[int, event_t] evn
#
#    cdef map[string, export_t] data
#
#    cdef map[string, vector[vector[int]]] pdgid
#    cdef map[string, vector[vector[vector[double]]]] pmu
#
#    cdef map[string, vector[vector[int]]] matched_bq
#    cdef map[string, vector[vector[int]]] matched_lp
#    
#    cdef map[string, vector[vector[double]]] ellipse 
#    cdef map[string, vector[vector[double]]] chi2_nu1
#    cdef map[string, vector[vector[double]]] chi2_nu2
#    cdef map[string, vector[vector[vector[double]]]] pmu_nu1
#    cdef map[string, vector[vector[vector[double]]]] pmu_nu2
#
