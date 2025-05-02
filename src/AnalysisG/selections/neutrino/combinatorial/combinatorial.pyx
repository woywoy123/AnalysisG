# distutils: language=c++
# cython: language_level=3
from cython.operator cimport dereference as dref
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *
from tqdm import tqdm

cdef extern from "<tools/merge_cast.h>" nogil:
    cdef void release_vector(vector[vector[double]]*)
    cdef void release_vector(vector[double]*)
    cdef void release_vector(vector[int]*)

cdef class Neutrino(ParticleTemplate):
    @property
    def matched_bquark(self): return int(self.nux.matched_bquark)
    @property
    def matched_lepton(self): return int(self.nux.matched_lepton)
    @property
    def ellipse(self): return self.nux.ellipse
    @property
    def chi2(self): return self.nux.chi2
 
cdef class Particle(ParticleTemplate):
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
            px.set_particle(ptx[v])
            self.TruthTops[env(name)][v % 2].append(px)

    cdef void assign_neutrinos(self, string name, vector[neutrino*] ptx):
        cdef int v
        cdef Neutrino px
        for v in range(ptx.size()):
            px = Neutrino()
            px.nux = ptx[v]
            px.set_particle(<particle_template*>(px.nux))
            self.RecoTops[env(name)][v % 2].append(px)

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

    def __len__(self): return self.lx

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













































