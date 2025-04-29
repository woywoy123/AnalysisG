# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.tools cimport *

cdef class Neutrino(ParticleTemplate):
    def __cinit__(self): self.nux = NULL
#    def __dealloc__(self): del self.nux

    @property
    def matched_bquark(self): return int(self.nux.matched_bquark)
    @property
    def matched_lepton(self): return int(self.nux.matched_lepton)
    @property
    def ellipse(self): return self.nux.ellipse
    @property
    def chi2(self): return self.nux.chi2
 
cdef class Particle(ParticleTemplate):
    def __cinit__(self): pass
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

cdef class Event:
    cdef public dict TruthTops
    cdef public dict RecoTops

    def __init__(self): 
        typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
        self.TruthTops = {i : [[], []] for i in typx}
        self.RecoTops  = {i : [[], []] for i in typx}

    cdef void assign_particles(self, string name, vector[particle_template*] ptx):
        cdef int v
        cdef Particle px
        for v in range(ptx.size()):
            px = Particle()
            px.ptr = ptx[v]
            self.TruthTops[env(name)][v % 2].append(px)

    cdef void assign_neutrinos(self, string name, vector[neutrino*] ptx):
        cdef int v
        cdef Neutrino px
        for v in range(ptx.size()):
            px = Neutrino()
            px.nux = ptx[v]
            px.ptr = <particle_template*>(px.nux)
            self.RecoTops[env(name)][v % 2].append(px)


cdef void loader(NuNuCombinatorial vl, tuple data):
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


cdef particle_template* make_particle(vector[double]* pmu, int pdgid):
    cdef particle_template* ptx = new particle_template()
    ptx.pt  = pmu.at(0); ptx.eta = pmu.at(1)
    ptx.phi = pmu.at(2); ptx.e   = pmu.at(3)
    ptx.pdgid = pdgid
    return ptx

cdef neutrino* make_neutrino(vector[double]* pmu, int lep, int bq, double elp, double chi):
    cdef neutrino* ptx = new neutrino()
    ptx.type = string(b"neutrino")
    ptx.pt  = pmu.at(0); ptx.eta = pmu.at(1)
    ptx.phi = pmu.at(2); ptx.e   = pmu.at(3)
    ptx.matched_bquark = bq; ptx.matched_lepton = lep
    ptx.ellipse = elp; ptx.chi2 = chi; 
    return ptx

cdef class NuNuCombinatorial(SelectionTemplate):
    def __dealloc__(self): del self.tt
    def __cinit__(self):
        typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
        attx = [
                "pmu", "matched_bquark", "matched_lepton", 
                "distance", "nu1_chi2", "nu2_chi2", 
                "nu1_pmu", "nu2_pmu", "pdgid"
        ]
        self.root_leaves = {i + "_" + j : loader for i in typx for j in attx}
        self.ptr = new combinatorial()
        self.tt = <combinatorial*>self.ptr
        self.ix = -1

    def Postprocessing(self):
        cdef vector[string] typx = [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]
        cdef string name 
        cdef int iy, iz

        cdef vector[vector[double]] vvd
        cdef vector[double] vd
        cdef vector[int] vi

        cdef vector[neutrino*] vnu
        cdef vector[particle_template*] vp
        cdef neutrino* ptx

        cdef Event ev = Event()
        for name in typx:
            self.lx = self.pmu[name].size()
            if self.ix < 0: return

            vvd = self.pmu[name][self.ix]
            vp.clear(); vp.clear()
            for iy in range(vvd.size()): 
                vp.push_back(make_particle(&vvd.at(iy), self.pdgid[name][self.ix][iy]))
            ev.assign_particles(name, vp)

            for iy in range(self.pmu_nu1[name][self.ix].size()):
                ptx = make_neutrino(
                        &self.pmu_nu1[name][self.ix][iy], 
                        self.matched_lp[name][self.ix][iy],
                        self.matched_bq[name][self.ix][iy],
                        self.ellipse[name][self.ix][iy], 
                        self.chi2_nu1[name][self.ix][iy]
                )
                vnu.push_back(ptx)

                ptx = make_neutrino(
                        &self.pmu_nu2[name][self.ix][iy], 
                        self.matched_lp[name][self.ix][iy],
                        self.matched_bq[name][self.ix][iy],
                        self.ellipse[name][self.ix][iy], 
                        self.chi2_nu2[name][self.ix][iy]
                )
                vnu.push_back(ptx)
            ev.assign_neutrinos(name, vnu)

        return ev

    def __iter__(self):
        self.ix = 0
        return self
    
    def __next__(self):
        if self.ix >= self.lx: raise StopIteration 
        cdef Event ev = self.Postprocessing()
        self.ix += 1
        return ev


    @property
    def NumDevice(self): return self.tt.num_device
    @NumDevice.setter
    def NumDevice(self, int val): self.tt.num_device = val


    @property
    def MassTop(self): return self.tt.masstop
    @MassTop.setter
    def MassTop(self, float val): self.tt.masstop = val

    @property
    def MassW(self): return self.tt.massw
    @MassW.setter
    def MassW(self, float val): self.tt.massw = val


