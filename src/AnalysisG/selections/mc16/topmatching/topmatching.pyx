# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef void fx_mass(TopMatching xl, tuple data): 
    cdef vector[float] val = <vector[float]>(data[1])
    if data[0] == "top_mass":              merge_data(&xl.top_mass             , &val); return 
    if data[0] == "topchildren_mass":      merge_data(&xl.topchildren_mass     , &val); return
    if data[0] == "toptruthjets_mass":     merge_data(&xl.toptruthjets_mass    , &val); return
    if data[0] == "topjets_children_mass": merge_data(&xl.topjets_children_mass, &val); return
    if data[0] == "topjets_leptons_mass":  merge_data(&xl.topjets_leptons_mass , &val); return

def fx_lep(TopMatching xl, tuple data):
    cdef vector[bool] val = <vector[bool]>(data[1])
    if data[0] == "topchildren_islep":      merge_data(&xl.topchildren_islep     , &val); return
    if data[0] == "toptruthjets_islep":     merge_data(&xl.toptruthjets_islep    , &val); return
    if data[0] == "topjets_children_islep": merge_data(&xl.topjets_children_islep, &val); return
    if data[0] == "topjets_leptons_islep":  merge_data(&xl.topjets_leptons_islep , &val); return

def fx_njx(TopMatching xl, tuple data):
    cdef vector[int] val = <vector[int]>(data[1])
    if data[0] == "toptruthjets_njets":     merge_data(&xl.toptruthjets_njets    , &val); return
    if data[0] == "topjets_leptons_njets":  merge_data(&xl.topjets_leptons_njets , &val); return
    if data[0] == "topjets_leptons_pdgid":  merge_data(&xl.topjets_leptons_pdgid , &val); return

cdef class TopMatching(SelectionTemplate):
    def __cinit__(self):
        self.root_leaves = {
                "top_mass"             : fx_mass,
                "topchildren_mass"     : fx_mass,
                "toptruthjets_mass"    : fx_mass,
                "topjets_children_mass": fx_mass,
                "topjets_leptons_mass" : fx_mass,

                "topchildren_islep"      : fx_lep, 
                "toptruthjets_islep"     : fx_lep,
                "topjets_children_islep" : fx_lep,
                "topjets_leptons_islep"  : fx_lep,

                "toptruthjets_njets"     : fx_njx,
                "topjets_leptons_njets"  : fx_njx,
                "topjets_leptons_pdgid"  : fx_njx
        }

        self.ptr = new topmatching()
        self.tt = <topmatching*>self.ptr

        self.n_truth_jets_lep = {}
        self.n_truth_jets_had = {}
        self.n_jets_lep = {}
        self.n_jets_had = {}

    def __dealloc__(self): del self.tt

    def Postprocessing(self):
        self.truth_top = self.top_mass

        self.truth_children  = {"all" : [], "lep" : [], "had" : []}
        self.truth_jets      = {"all" : [], "lep" : [], "had" : []}   
        self.jets_truth_leps = {"all" : [], "lep" : [], "had" : []}
        self.jet_leps        = {"all" : [], "lep" : [], "had" : []}

        self.truth_children["all"]  = self.topchildren_mass
        self.truth_jets["all"]      = self.toptruthjets_mass
        self.jets_truth_leps["all"] = self.topjets_children_mass
        self.jet_leps["all"]       = self.topjets_leptons_mass
      
        cdef str lx
        cdef float mx
        cdef dict mode
        cdef int idx, pdg, nj
        for idx in range(self.topchildren_islep.size()):
            lx = "lep" if self.topchildren_islep[idx] else "had"
            mx = self.topchildren_mass[idx]
            self.truth_children[lx] += [mx]
            
        for idx in range(self.toptruthjets_islep.size()):
            lx = "lep" if self.toptruthjets_islep[idx] else "had"
            mx = self.toptruthjets_mass[idx]
            self.truth_jets[lx] += [mx]

            nj = self.toptruthjets_njets[idx]
            lx = str(nj) + " - Truth Jets"
            mode = self.n_truth_jets_lep if self.toptruthjets_islep[idx] else self.n_truth_jets_had
            if lx not in mode: mode[lx] = []
            mode[lx] += [mx]

        for idx in range(self.topjets_children_islep.size()):
            lx = "lep" if self.topjets_children_islep[idx] else "had"
            mx = self.topjets_children_mass[idx]
            self.jets_truth_leps[lx] += [mx]
 
        for idx in range(self.topjets_leptons_islep.size()):
            lx  = "lep" if self.topjets_leptons_islep[idx] else "had"
            mx = self.topjets_leptons_mass[idx]
            nj = self.toptruthjets_njets[idx]
            self.jet_leps[lx] += [mx]

            nj = self.topjets_leptons_njets[idx]
            lx = str(nj) + " - Jets"
            mode = self.n_jets_lep if self.topjets_leptons_islep[idx] else self.n_jets_had
            if lx not in mode: mode[lx] = []
            mode[lx] += [mx]

            if not self.topjets_leptons_islep[idx]: continue
            pdg = self.topjets_leptons_pdgid[idx] 
            lx = "lep-" + (str(pdg) if pdg else "miss")
            if lx not in self.jet_leps: self.jet_leps[lx] = []
            self.jet_leps[lx] += [mx]

    cdef void transform_dict_keys(self): pass
