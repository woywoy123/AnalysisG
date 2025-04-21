# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "validation.h":
    cdef cppclass validation(selection_template):
        validation() except +
        int num_device

cdef cppclass neutrino(particle_template):
    double distance

cdef cppclass event:
    double met
    double phi

    vector[particle_template*] truth_neutrino
    map[string, vector[vector[neutrino*]]] dynamic_neutrino
    map[string, vector[vector[neutrino*]]] static_neutrino
    map[string, vector[particle_template*]] particles

cdef inline particle_template* make_particle(vector[double]* pmu, int pdg) noexcept nogil:
    if not pmu.size(): return NULL
    cdef particle_template* p = new particle_template()
    p.type = string(b"particle")
    p.pt  = pmu.at(0)
    p.eta = pmu.at(1)
    p.phi = pmu.at(2)
    p.e   = pmu.at(3)
    p.pdgid = pdg
    return p

cdef inline neutrino* make_neutrino(vector[double]* pmu, double dst) noexcept nogil:
    if not pmu.size(): return NULL
    cdef neutrino* p = new neutrino()
    p.type = string(b"particle")
    p.pt  = pmu.at(0)
    p.eta = pmu.at(1)
    p.phi = pmu.at(2)
    p.e   = pmu.at(3)
    p.distance = dst
    return p

cdef class Validation(SelectionTemplate):
    cdef validation* tt
    cdef public list Events

    cdef map[string, vector[vector[vector[double]]]] nu1_static
    cdef map[string, vector[vector[vector[double]]]] nu2_static
    cdef map[string, vector[vector[double]]]   static_distances

    cdef map[string, vector[vector[vector[double]]]] nu1_dynamic
    cdef map[string, vector[vector[vector[double]]]] nu2_dynamic
    cdef map[string, vector[vector[double]]]   dynamic_distances

    cdef map[string, vector[vector[vector[double]]]] pmu
    cdef map[string, vector[vector[int]]] pdgid

    cdef vector[double] met
    cdef vector[double] phi



cdef inline void loader(Validation vl, tuple data):
    cdef str name = data[0]
    cdef bool is_pmc  = "pmu"     in name
    cdef bool is_stat = "static"  in name
    cdef bool is_dyn  = "dynamic" in name
    cdef bool is_nu1  = "nu1"     in name
    cdef bool is_nu2  = "nu2"     in name
    cdef bool is_dst  = "dst"     in name

    cdef string base_name
    if "top_children"  in name: base_name = b"top_children"
    elif "truthjet"    in name: base_name = b"truthjet"
    elif "jetchildren" in name: base_name = b"jetchildren"
    elif "jetleptons"  in name: base_name = b"jetleptons"
    elif "met"         in name: vl.met.push_back(data[1]); return 
    elif "phi"         in name: vl.phi.push_back(data[1]); return 
    else: return 

    if is_pmc and not is_nu1 and not is_nu2:  vl.pmu[base_name].push_back(<vector[vector[double]]>(data[1]));         return
    if is_pmc and is_nu1 and is_stat:         vl.nu1_static[base_name].push_back(<vector[vector[double]]>(data[1]));  return
    if is_pmc and is_nu2 and is_stat:         vl.nu2_static[base_name].push_back(<vector[vector[double]]>(data[1]));  return
    if is_dst and is_stat:                    vl.static_distances[base_name].push_back(<vector[double]>(data[1]));    return
    if is_pmc and is_nu1 and is_dyn:          vl.nu1_dynamic[base_name].push_back(<vector[vector[double]]>(data[1])); return
    if is_pmc and is_nu2 and is_dyn:          vl.nu2_dynamic[base_name].push_back(<vector[vector[double]]>(data[1])); return
    if is_dst and is_dyn:                     vl.dynamic_distances[base_name].push_back(<vector[double]>(data[1]));   return
    vl.pdgid[base_name].push_back(<vector[int]>(data[1]))






