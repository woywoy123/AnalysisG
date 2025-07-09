# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *

cdef missing(MissingET zp, tuple data):
    cdef str name
    name, val = data
    zp.indx[enc(name)] += 1
    if zp.indx[enc(name)] > zp.data.size(): zp.data.push_back(container_t())

    cdef container_t* pkl = &zp.data[zp.data.size()-1]
    if name == "missing_event_px"     : pkl.missing_evn_px   = <double>(val)                ; return;       
    if name == "missing_event_py"     : pkl.missing_evn_py   = <double>(val)                ; return;       
    if name == "missing_detector_px"  : pkl.missing_det_px   = <double>(val)                ; return;       
    if name == "missing_detector_py"  : pkl.missing_det_py   = <double>(val)                ; return;       
    if name == "missing_detector_pz"  : pkl.missing_det_pz   = <double>(val)                ; return;       
    if name == "missing_neutrino_px"  : pkl.missing_nus_px   = <double>(val)                ; return;       
    if name == "missing_neutrino_py"  : pkl.missing_nus_py   = <double>(val)                ; return;       
    if name == "missing_neutrino_pz"  : pkl.missing_nus_pz   = <double>(val)                ; return;       
    if name == "num_neutrino"         : pkl.num_neutrino     = <double>(val)                ; return;       
    if name == "num_leptons"          : pkl.num_leptons      = <double>(val)                ; return;       
    if name == "num_leptons_reco"     : pkl.num_leptons_reco = <double>(val)                ; return;       
    if name == "top_index"            : pkl.top_index        = <vector[int]>(val)           ; return;       
    if name == "mass_tru_top"         : pkl.mass_tru_top     = <vector[double]>(val)        ; return;       
    if name == "mass_tru_top_blnu"    : pkl.mass_tru_top3    = <vector[double]>(val)        ; return;       
    if name == "truth_neutrino_pmc"   : pkl.tru_nu           = <vector[vector[double]]>(val); return;       
    if name == "chi2_solutions"       : pkl.chi2_sols        = <vector[double]>(val)        ; return;       
    if name == "top_index_solutions_1": pkl.top_index_sols1  = <vector[int]>(val)           ; return;       
    if name == "top_index_solutions_2": pkl.top_index_sols2  = <vector[int]>(val)           ; return;       
    if name == "top_mass_solutions_1" : pkl.top_mass_sols1   = <vector[double]>(val)        ; return;       
    if name == "top_mass_solutions_2" : pkl.top_mass_sols2   = <vector[double]>(val)        ; return;       
    if name == "reco_neutrino_1_pmc"  : pkl.nu1              = <vector[vector[double]]>(val); return;       
    if name == "reco_neutrino_2_pmc"  : pkl.nu2              = <vector[vector[double]]>(val); return;       
    if name == "reco_plane_w1"        : pkl.agnR_pln_w1      = <vector[vector[double]]>(val); return;       
    if name == "reco_plane_w2"        : pkl.agnR_pln_w2      = <vector[vector[double]]>(val); return;       
    if name == "reco_plane_t1"        : pkl.agnR_pln_t1      = <vector[vector[double]]>(val); return;       
    if name == "reco_plane_t2"        : pkl.agnR_pln_t2      = <vector[vector[double]]>(val); return;       
    if name == "reco_plane_t1xt2"     : pkl.agnR_pln_t1xt2   = <vector[vector[double]]>(val); return;       
    if name == "reco_dR_nu_lep1"      : pkl.agnR_dR_nu_lep1  = <vector[double]>(val)        ; return;       
    if name == "reco_dR_nu_lep2"      : pkl.agnR_dR_nu_lep2  = <vector[double]>(val)        ; return;       
    if name == "reco_has_null"        : pkl.agnR_has_null    = <vector[bool]>(val)          ; return;       
    if name == "truth_plane_w1"       : pkl.agnT_pln_w1      = <vector[vector[double]]>(val); return;       
    if name == "truth_plane_w2"       : pkl.agnT_pln_w2      = <vector[vector[double]]>(val); return;       
    if name == "truth_plane_t1"       : pkl.agnT_pln_t1      = <vector[vector[double]]>(val); return;       
    if name == "truth_plane_t2"       : pkl.agnT_pln_t2      = <vector[vector[double]]>(val); return;       
    if name == "truth_plane_t1xt2"    : pkl.agnT_pln_t1xt2   = <vector[vector[double]]>(val); return;       
    if name == "truth_dR_nu_lep1"     : pkl.agnT_dR_nu_lep1  = <vector[double]>(val)        ; return;       
    if name == "truth_dR_nu_lep2"     : pkl.agnT_dR_nu_lep2  = <vector[double]>(val)        ; return;       
    if name == "truth_has_null"       : pkl.agnT_has_null    = <vector[bool]>(val)          ; return;       

cdef class MissingET(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new met()
        self.tt = <met*>self.ptr
        self.root_leaves = {
                "missing_event_px"      : missing, 
                "missing_event_py"      : missing, 
                "missing_detector_px"   : missing, 
                "missing_detector_py"   : missing, 
                "missing_detector_pz"   : missing, 
                "missing_neutrino_px"   : missing, 
                "missing_neutrino_py"   : missing, 
                "missing_neutrino_pz"   : missing, 
                "num_neutrino"          : missing, 
                "num_leptons"           : missing, 
                "num_leptons_reco"      : missing, 
                "top_index"             : missing, 
                "mass_tru_top"          : missing, 
                "mass_tru_top_blnu"     : missing, 
                "truth_neutrino_pmc"    : missing, 
                "chi2_solutions"        : missing, 
                "top_index_solutions_1" : missing, 
                "top_index_solutions_2" : missing, 
                "top_mass_solutions_1"  : missing, 
                "top_mass_solutions_2"  : missing, 
                "reco_neutrino_1_pmc"   : missing, 
                "reco_neutrino_2_pmc"   : missing, 
                "reco_plane_w1"         : missing, 
                "reco_plane_w2"         : missing, 
                "reco_plane_t1"         : missing, 
                "reco_plane_t2"         : missing, 
                "reco_plane_t1xt2"      : missing, 
                "reco_dR_nu_lep1"       : missing, 
                "reco_dR_nu_lep2"       : missing, 
                "reco_has_null"         : missing, 
                "truth_plane_w1"        : missing, 
                "truth_plane_w2"        : missing, 
                "truth_plane_t1"        : missing, 
                "truth_plane_t2"        : missing, 
                "truth_plane_t1xt2"     : missing, 
                "truth_dR_nu_lep1"      : missing, 
                "truth_dR_nu_lep2"      : missing, 
                "truth_has_null"        : missing, 
        }   

    def __dealloc__(self): del self.tt
    @property
    def masstop(self): return self.tt.masstop
    @masstop.setter
    def masstop(self, val): self.tt.masstop = val

    @property
    def massw(self): return self.tt.massw
    @massw.setter
    def massw(self, val): self.tt.massw = val

    @property
    def perturb(self): return self.tt.perturb
    @perturb.setter
    def perturb(self, val): self.tt.perturb = val

    @property
    def distance(self): return self.tt.distance
    @distance.setter
    def distance(self, val): self.tt.distance = val

    @property
    def steps(self): return self.tt.steps
    @steps.setter
    def steps(self, val): self.tt.steps = val

