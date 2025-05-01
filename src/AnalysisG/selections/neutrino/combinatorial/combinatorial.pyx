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

#cdef class Particle(ParticleTemplate):
#    cdef sets ptx
#    cdef public double ellipse 
#    cdef public double chi2 
#    cdef public int matched_bquark
#    cdef public int matched_lepton
#
#    cdef Particle release(self, particle_template* ptf):
#        cdef Particle px = Particle()
#        px.set_particle(ptf)
#        return px
#
#    cdef Particle release_nu(self, neutrino* ptf):
#        cdef Particle px = Particle()
#        px.set_particle(<particle_template*>(ptf))
#        px.ellipse        = ptf.ellipse       
#        px.chi2           = ptf.chi2          
#        px.matched_bquark = ptf.matched_bquark
#        px.matched_lepton = ptf.matched_lepton
#        return px
#
#    @property
#    def TruthNeutrino(self):
#        if self.ptx.tru_nu == NULL: return None
#        return self.release(self.ptx.tru_nu)
#
#    @property
#    def RecoNeutrino(self):
#        if self.ptx.rec_nu == NULL: return None
#        return self.release_nu(self.ptx.rec_nu)
#
#    @property
#    def BQuark(self):
#        if self.ptx.tru_b == NULL: return None
#        return self.release(self.ptx.tru_b)
#
#    @property
#    def Lepton(self):
#        if self.ptx.tru_l == NULL: return None
#        return self.release(self.ptx.tru_l)
#
#    @property
#    def RecoW(self):
#        if self.ptx.rec_wboson == NULL: return None
#        return self.release(self.ptx.rec_wboson)
#
#    @property
#    def RecoTop(self):
#        if self.ptx.rec_top == NULL: return None
#        return self.release(self.ptx.rec_top)
#
#    @property
#    def TruthW(self):
#        if self.ptx.tru_wboson == NULL: return None
#        return self.release(self.ptx.tru_wboson)
#
#    @property
#    def TruthTop(self):
#        if self.ptx.tru_top == NULL: return None
#        return self.release(self.ptx.tru_top)
# 
#cdef class Export:
#    cdef export_t data
#    cdef dict make_data(self, map[int, map[string, vector[sets]]]* ox):
#        cdef dict out = {}
#        try: ox.at(0).size()
#        except IndexError: return out
#
#        cdef int ix
#        cdef str pairs
#        cdef Particle pto
#        cdef pair[string, vector[sets]] itx
#        for itx in dref(ox)[0]:
#            pairs = env(itx.first)
#            out[pairs] = {0 : [], 1 : []}
#            for ix in range(itx.second.size()):
#                pto = Particle()
#                pto.ptx = itx.second.at(ix)
#                out[pairs][0].append(pto)
#
#                pto = Particle()
#                pto.ptx = ox.at(1).at(itx.first).at(ix)
#                out[pairs][1].append(pto)
#        return out
#
#    @property
#    def n_correct(self): return as_dict(&self.data.n_correct)
#    @property
#    def n_b_swapped(self): return as_dict(&self.data.n_b_swapped)
#    @property
#    def n_l_swapped(self): return as_dict(&self.data.n_l_swapped)
#    @property
#    def n_bl_swapped(self): return as_dict(&self.data.n_bl_swapped)
#    @property
#    def n_unmatched(self): return as_dict(&self.data.n_unmatched)
#    @property
#    def n_non_nunu(self): return as_dict(&self.data.n_non_nunu)
#    @property
#    def n_tru_nu(self): return as_dict(&self.data.n_tru_nu) 
#    @property
#    def n_rec_nu(self): return as_dict(&self.data.n_rec_nu)   
#    @property
#    def num_sols(self): return as_dict(&self.data.num_sols)   
#    @property
#    def merged_jet(self): return as_dict(&self.data.merged_jet)
#    @property
#    def correct(self): return self.make_data(&self.data.correct)
#    @property
#    def swapped_bs(self): return self.make_data(&self.data.swapped_bs)
#    @property
#    def swapped_ls(self): return self.make_data(&self.data.swapped_ls)
#    @property
#    def swapped_bl(self): return self.make_data(&self.data.swapped_bl)
#    @property
#    def fake(self): return self.make_data(&self.data.fake_nus)
# 
#
#
#cdef class NuNuCombinatorial(SelectionTemplate):
#    def __dealloc__(self): 
#        cdef string name 
#        cdef pair[int, event_t] itx
#        for name in [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]:
#            for itx in self.evn:
#                delete(&itx.second.truth_tops[name])
#                delete(&itx.second.reco_tops[name])
#        del self.tt
#
#    def __cinit__(self):
#        typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
#        attx = [
#                "pmu", "matched_bquark", "matched_lepton", 
#                "distance", "nu1_chi2", "nu2_chi2",
#                "nu1_pmu", "nu2_pmu", "pdgid"
#        ]
#        self.root_leaves = {i + "_" + j : loader for i in typx for j in attx}
#        self.ptr = new combinatorial()
#        self.tt = <combinatorial*>self.ptr
#        cdef map[string, vector[atomic]] v1, v2
#        self.con = container(v1, v2, self.ptr)
#
#    def Postprocessing(self):
#        cdef int ix
#        cdef string name 
#        cdef vector[particle_template*] vp
#        cdef vector[neutrino*] vnu1, vnu2
#        cdef vector[string] typx = [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]
#        for name in typx:
#            self.con.atomics[name]; self.con.chi2_atomics[name]
#
#            for ix in tqdm(range(self.pmu[name].size()), desc = env(name)):
#                vp.clear(); vnu1.clear(); vnu2.clear()
#                self.evn[ix].truth_tops[name][0] = vp
#                self.evn[ix].truth_tops[name][1] = vp
#                self.evn[ix].reco_tops[name][0]  = vp
#                self.evn[ix].reco_tops[name][1]  = vp
#
#                vp = make_particle(&self.pmu[name][ix], &self.pdgid[name][ix])
#                vnu1 = make_neutrino(
#                        &self.pmu_nu1[name][ix]   , &self.matched_lp[name][ix], 
#                        &self.matched_bq[name][ix], &self.ellipse[name][ix], 
#                        &self.chi2_nu1[name][ix]
#                )
#                vnu2 = make_neutrino(
#                        &self.pmu_nu2[name][ix], &self.matched_lp[name][ix], 
#                        &self.matched_bq[name][ix], &self.ellipse[name][ix], 
#                        &self.chi2_nu2[name][ix]
#                )
#
#                self.matched_lp[name][ix].clear();  release_vector(&self.matched_lp[name][ix])
#                self.matched_bq[name][ix].clear();  release_vector(&self.matched_bq[name][ix])
#                self.ellipse[name][ix].clear();     release_vector(&self.ellipse[name][ix])
#
#                assign_particles(name, &vp, &self.evn[ix])
#                assign_neutrinos(name, &vnu1, &vnu2, &self.evn[ix])
#                add_container(name, &self.con, &self.evn[ix], False)
#                add_container(name, &self.con, &self.evn[ix],  True)
#
#            self.data[name + b"-nominal"]  = get_export(name, &self.con, False)
#            self.data[name + b"-cx2match"] = get_export(name, &self.con,  True)
#
#    def get(self, str name, bool chi):
#        cdef string nx = enc(name) + (b"-cx2match" if chi else b"-nominal")
#        if not self.data.count(nx): return False
#        cdef Export exp = Export()
#        exp.data = self.data[nx]
#        return exp
#
#    @property
#    def NumDevice(self): return self.tt.num_device
#    @NumDevice.setter
#    def NumDevice(self, int val): self.tt.num_device = val
#
#    @property
#    def MassTop(self): return self.tt.masstop
#    @MassTop.setter
#    def MassTop(self, float val): self.tt.masstop = val
#
#    @property
#    def MassW(self): return self.tt.massw
#    @MassW.setter
#    def MassW(self, float val): self.tt.massw = val

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













































