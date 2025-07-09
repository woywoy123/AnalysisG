# distutils: language=c++
# cython: language_level=3

import vector as vxc
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *
from cython.parallel cimport prange

cdef class Neutrino(ParticleTemplate):
    cdef neutrino* ptx
    cdef public double chi2
    def __cinit__(self): self.ptx = NULL
#    def __dealloc__(self): del self.ptx
    
    @property
    def distance(self): 
        if self.ptx == NULL: 
            self.ptx = new neutrino()
            self.ptr = <particle_template*>(self.ptx)
        return self.ptx.distance

    @distance.setter
    def distance(self, double v):
        if self.ptx == NULL: 
            self.ptx = new neutrino()
            self.ptr = <particle_template*>(self.ptx)
        self.ptx.distance = v

    @property
    def vec(self): return vxc.obj(**{"px" : self.px, "py" : self.py, "pz" : self.pz, "energy" : self.e})

cdef class Particle(ParticleTemplate):
#    def __dealloc__(self): del self.ptr

    @property
    def vec(self): return vxc.obj(**{"px" : self.px, "py" : self.py, "pz" : self.pz, "energy" : self.e})

cdef class Event:
    cdef event* ptr
    cdef public list TruthNeutrinos  
    cdef public dict DynamicNeutrino 
    cdef public dict StaticNeutrino  
    cdef public dict Particles       

    def __dealloc__(self): del self.ptr
    def __cinit__(self): self.ptr = NULL

    def __init__(self):
        typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
        self.TruthNeutrinos  = []
        self.DynamicNeutrino = {i : {0 : [], 1 : []} for i in typx}
        self.StaticNeutrino  = {i : {0 : [], 1 : []} for i in typx}
        self.Particles       = {i : {0 : [], 1 : []} for i in typx}

    @property
    def met(self): return self.ptr.met
    @property
    def phi(self): return self.ptr.phi
    @property
    def vec(self): return vxc.obj(pt = self.met, phi = self.phi)

    cdef build(self):
        cdef int i, t
        cdef string base
        cdef Neutrino nu
        cdef Particle prt

        for i in range(self.ptr.truth_neutrino.size()):
            prt = Particle()
            prt.ptr = self.ptr.truth_neutrino[i]
            self.TruthNeutrinos.append(prt)
        
        for k in ["top_children", "truthjet", "jetchildren", "jetleptons"]:
            base = enc(k)
            for i in range(self.ptr.dynamic_neutrino[base].size()):
                for t in range(2):
                    nu = Neutrino()
                    nu.ptx = self.ptr.dynamic_neutrino[base][i][t]
                    nu.ptr = <particle_template*>(nu.ptx)
                    self.DynamicNeutrino[k][t].append(nu)

            for i in range(self.ptr.static_neutrino[base].size()):
                for t in range(2):
                    nu = Neutrino()
                    nu.ptx = self.ptr.static_neutrino[base][i][t]
                    nu.ptr = <particle_template*>(nu.ptx)
                    self.StaticNeutrino[k][t].append(nu)

            for i in range(self.ptr.particles[base].size()):
                t = 1 if i > 1 else 0
                prt = Particle()
                prt.ptr = self.ptr.particles[base][i]
                self.Particles[k][t].append(prt)

cdef class Validation(SelectionTemplate):
    def __dealloc__(self): del self.tt
    def __cinit__(self):
        self.Events = []
        typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
        attrs = [
                "pmu", "pdgid", 
                "dynamic_nu1_pmu", "dynamic_nu2_pmu", "dynamic_dst",
                 "static_nu1_pmu",  "static_nu2_pmu",  "static_dst"
        ] 
        xp = [i + "_" + j for i in typx for j in attrs] + ["met", "phi"]
        self.root_leaves = {i : loader for i in xp}
        self.ptr = new validation()
        self.tt = <validation*>self.ptr

    @property
    def NumDevices(self): return self.tt.num_device
    @NumDevices.setter
    def NumDevices(self, int vl): self.tt.num_device = vl

    def Postprocessing(self):
        cdef int i, k
        cdef bool ix
        cdef vector[event*] evnts
        cdef vector[double]* met = &self.met
        cdef vector[double]* phi = &self.phi

        cdef map[string, vector[vector[vector[double]]]]* nu1_s = &self.nu1_static
        cdef map[string, vector[vector[vector[double]]]]* nu2_s = &self.nu2_static
        cdef map[string, vector[vector[vector[double]]]]* nu1_d = &self.nu1_dynamic 
        cdef map[string, vector[vector[vector[double]]]]* nu2_d = &self.nu2_dynamic
        cdef map[string, vector[vector[vector[double]]]]* pmu   = &self.pmu  

        cdef map[string, vector[vector[int]]]*    pdgid   = &self.pdgid
        cdef map[string, vector[vector[double]]]* dyn_dst = &self.dynamic_distances 
        cdef map[string, vector[vector[double]]]* sta_dst = &self.static_distances
        cdef vector[string] names = [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]

        cdef string base
        cdef event* otx
        cdef particle_template* ptx
        cdef vector[neutrino*] oxm

        for i in prange(self.met.size(), nogil = True):
            otx = new event()
            otx.met = met.at(i)
            otx.phi = phi.at(i)
            evnts.push_back(otx)

            for base in names:
                for k in range(pmu.at(base).at(i).size()):
                    ptx = make_particle(&pmu.at(base).at(i).at(k), pdgid.at(base).at(i).at(k))
                    if ptx is NULL: continue 
                    ix = ptx.is_nu 
                    if ix: otx.truth_neutrino.push_back(ptx)
                    else: otx.particles[base].push_back(ptx)
          

                for k in range(nu1_s.at(base).at(i).size()): 
                    oxm.clear()
                    oxm.push_back(make_neutrino(&nu1_s.at(base).at(i).at(k), sta_dst.at(base).at(i).at(k)))
                    oxm.push_back(make_neutrino(&nu2_s.at(base).at(i).at(k), sta_dst.at(base).at(i).at(k)))
                    otx.static_neutrino[base].push_back(oxm)

                for k in range(nu1_d.at(base).at(i).size()): 
                    oxm.clear()
                    oxm.push_back(make_neutrino(&nu1_d.at(base).at(i).at(k), dyn_dst.at(base).at(i).at(k)))
                    oxm.push_back(make_neutrino(&nu2_d.at(base).at(i).at(k), dyn_dst.at(base).at(i).at(k)))
                    otx.dynamic_neutrino[base].push_back(oxm)

        nu1_s.clear(); nu2_s.clear(); nu1_d.clear(); nu2_d.clear()
        pmu.clear(); pdgid.clear(); dyn_dst.clear(); sta_dst.clear()

        cdef Event ev
        for i in range(evnts.size()):
            ev = Event()
            ev.ptr = evnts[i]
            ev.build()
            self.Events.append(ev)
