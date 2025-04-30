# distutils: language=c++
# cython: language_level=3
from tqdm import tqdm
from AnalysisG.core.tools cimport *
from AnalysisG.core.particle_template cimport *

cdef extern from "<tools/merge_cast.h>" nogil:
    cdef void release_vector(vector[vector[double]]*)
    cdef void release_vector(vector[double]*)
    cdef void release_vector(vector[int]*)

cdef class Particle(ParticleTemplate):
    pass


cdef class Export:
    cdef export_t* data
    def __cinit__(self): pass
    def __init__(self): pass

    @property
    def n_correct(self): return as_dict(&self.data.n_correct)
    @property
    def n_b_swapped(self): return as_dict(&self.data.n_b_swapped)
    @property
    def n_l_swapped(self): return as_dict(&self.data.n_l_swapped)
    @property
    def n_bl_swapped(self): return as_dict(&self.data.n_bl_swapped)
    @property
    def n_unmatched(self): return as_dict(&self.data.n_unmatched)
    @property
    def n_non_nunu(self): return as_dict(&self.data.n_non_nunu)
    @property
    def n_tru_nu(self): return as_dict(&self.data.n_tru_nu) 
    @property
    def n_rec_nu(self): return as_dict(&self.data.n_rec_nu)   
    @property
    def num_sols(self): return as_dict(&self.data.num_sols)   
    @property
    def merged_jet(self): return as_dict(&self.data.merged_jet)
   
   #map[int, map[string, vector[sets]]] correct       
   #map[int, map[string, vector[sets]]] swapped_bs 
   #map[int, map[string, vector[sets]]] swapped_bl 
   #map[int, map[string, vector[sets]]] swapped_ls 
   #map[int, map[string, vector[sets]]] fake_nus   




cdef class NuNuCombinatorial(SelectionTemplate):
    def __dealloc__(self): 
        cdef string name 
        cdef pair[int, event_t] itx
        for name in [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]:
            for itx in self.evn:
                delete(&itx.second.truth_tops[name])
                delete(&itx.second.reco_tops[name])
        del self.tt

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
        cdef map[string, vector[atomic]] v1, v2
        self.con = container(v1, v2, self.ptr)

    def Postprocessing(self):
        cdef int ix
        cdef string name 
        cdef vector[particle_template*] vp
        cdef vector[neutrino*] vnu1, vnu2
        cdef vector[string] typx = [b"top_children", b"truthjet", b"jetchildren", b"jetleptons"]

        for name in typx:
            self.con.atomics[name]; self.con.chi2_atomics[name]

            for ix in tqdm(range(self.pmu[name].size()), desc = env(name)):
                vp.clear(); vnu1.clear(); vnu2.clear()
                self.evn[ix].truth_tops[name][0] = vp
                self.evn[ix].truth_tops[name][1] = vp
                self.evn[ix].reco_tops[name][0]  = vp
                self.evn[ix].reco_tops[name][1]  = vp

                vp = make_particle(&self.pmu[name][ix], &self.pdgid[name][ix])
                vnu1 = make_neutrino(
                        &self.pmu_nu1[name][ix]   , &self.matched_lp[name][ix], 
                        &self.matched_bq[name][ix], &self.ellipse[name][ix], 
                        &self.chi2_nu1[name][ix]
                )
                vnu2 = make_neutrino(
                        &self.pmu_nu2[name][ix], &self.matched_lp[name][ix], 
                        &self.matched_bq[name][ix], &self.ellipse[name][ix], 
                        &self.chi2_nu2[name][ix]
                )

                self.matched_lp[name][ix].clear();  release_vector(&self.matched_lp[name][ix])
                self.matched_bq[name][ix].clear();  release_vector(&self.matched_bq[name][ix])
                self.ellipse[name][ix].clear();     release_vector(&self.ellipse[name][ix])

                assign_particles(name, &vp, &self.evn[ix])
                assign_neutrinos(name, &vnu1, &vnu2, &self.evn[ix])
                add_container(name, &self.con, &self.evn[ix], False)
                add_container(name, &self.con, &self.evn[ix],  True)

            self.data[name + b"-nominal"] = get_export(name, &self.con, False)
            self.data[name + b"-cx2match"] = get_export(name, &self.con,  True)

    def get(self, str name, bool chi):
        cdef string nx = enc(name) + (b"-cx2match" if chi else b"-nominal")
        if not self.data.count(nx): return False
        cdef Export exp = Export()
        exp.data = &self.data[nx]
        return exp

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


