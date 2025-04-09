# distutils: language=c++
# cython: language_level=3

from libcpp cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.core.selection_template cimport *

cdef str pdgid(int ipt):
    if abs(ipt) == 1 : return "d" 
    if abs(ipt) == 2 : return "u" 
    if abs(ipt) == 3 : return "s" 
    if abs(ipt) == 4 : return "c" 
    if abs(ipt) == 5 : return "b" 
    if abs(ipt) == 6 : return "t"
    if abs(ipt) == 11: return "e" 
    if abs(ipt) == 12: return "$\\nu_{e}$" 
    if abs(ipt) == 13: return "$\\mu$" 
    if abs(ipt) == 14: return "$\\nu_{\\mu}$" 
    if abs(ipt) == 15: return "$\\tau$" 
    if abs(ipt) == 16: return "$\\nu_{\\tau}$"
    if abs(ipt) == 21: return "g" 
    if abs(ipt) == 22: return "$\\gamma$"
    return "n/a"


cdef void merge(vector[float]* ix, vector[float]* iy):
    cdef int i
    for i in range(iy.size()): ix.push_back(iy.at(i))

def res(ck, data):
    cdef string name = enc(data[0].split("_")[-1])
    cdef vector[float] vx = <vector[float]>(data[1])
    cdef ChildrenKinematics px = ck
    merge(&px.r_data[name], &vx)

def spc(ck, data):
    cdef string name = enc(data[0].split("_")[-1])
    cdef vector[float] vx = <vector[float]>(data[1])
    cdef ChildrenKinematics px = ck
    merge(&px.s_data[name], &vx)

def rdcy(ck, data):
    cdef str xd = "frc." if "frc" in data[0] else ""
    cdef string name = enc(xd + data[0].split("_")[-1])
    cdef vector[float] vx = <vector[float]>(data[1])
    cdef ChildrenKinematics px = ck
    merge(&px.r_decay[name], &vx)

def sdcy(ck, data):
    cdef str xd = "frc." if "frc" in data[0] else ""
    cdef string name = enc(xd + data[0].split("_")[-1])
    cdef vector[float] vx = <vector[float]>(data[1])
    cdef ChildrenKinematics px = ck
    merge(&px.s_decay[name], &vx)

def tpm(ck, data):
    _name, val = data
    cdef string name = enc(_name.split("_")[-1])
    cdef vector[float] vx = <vector[float]>(val)
    cdef ChildrenKinematics px = ck
    merge(&px.top_pem[name], &vx)


cdef class ChildrenKinematics(SelectionTemplate):
    def __cinit__(self):
        self.root_leaves = {
                "res_pt"     : res, "spec_pt"     : spc,  
                "res_energy" : res, "spec_energy" : spc, 
                "res_eta"    : res, "spec_eta"    : spc,  
                "res_phi"    : res, "spec_phi"    : spc, 
               "res_pdgid"   : res, "spec_pdgid"   : spc, 

          "res_decay_pt"     : rdcy, "spec_decay_pt"     : sdcy,  
          "res_decay_energy" : rdcy, "spec_decay_energy" : sdcy, 
          "res_decay_eta"    : rdcy, "spec_decay_eta"    : sdcy,  
          "res_decay_phi"    : rdcy, "spec_decay_phi"    : sdcy, 

          "res_decay_pdgid"  : rdcy, "spec_decay_pdgid"  : sdcy, 
          "res_decay_islep"  : rdcy, "spec_decay_islep"  : sdcy, 
          "res_decay_mass"   : rdcy, "spec_decay_mass"   : sdcy, 
          "res_decay_dR"     : rdcy, "spec_decay_dR"     : sdcy, 
          "res_decay_frc_e"  : rdcy, "spec_decay_frc_e"  : sdcy, 
          "res_decay_frc_pt" : rdcy, "spec_decay_frc_pt" : sdcy, 

          "top_perm_pt"   : tpm, "top_perm_energy" : tpm, 
          "top_perm_mass" : tpm, "top_perm_dR"     : tpm, 
          "top_perm_RR"   : tpm, "top_perm_SS"     : tpm,
          "top_perm_RS"   : tpm, "top_perm_CT"     : tpm, 
          "top_perm_FT"   : tpm
        }                            

        self.res_kinematics        = {}
        self.spec_kinematics       = {}
        self.res_pdgid_kinematics  = {}
        self.spec_pdgid_kinematics = {}
        
        self.res_decay_mode        = {}
        self.spec_decay_mode       = {}
        self.fractional            = {}
        
        self.dr_clustering         = {}
        self.top_children_dr       = {}

        self.ptr = new childrenkinematics()
        self.tt = <childrenkinematics*>self.ptr

    def __dealloc__(self): del self.tt

    def Postprocessing(self):
        cdef int i
        cdef str p, k, lx
        cdef vector[float] pdg, dxm
        cdef list kx = ["pt", "eta", "phi", "energy"] 
        for p in kx: self.res_kinematics[p]  = self.r_data[enc(p)]
        for p in kx: self.spec_kinematics[p]  = self.s_data[enc(p)]

        pdg = self.r_data[b"pdgid"]
        for i in range(pdg.size()): 
            if pdg[i] == 0: continue
            p = pdgid(int(pdg[i]))
            if p not in self.res_pdgid_kinematics: self.res_pdgid_kinematics[p] = {k : [] for k in kx}
            for k in kx: self.res_pdgid_kinematics[p][k] += [self.r_data[enc(k)][i]]
        self.r_data.clear() 

        pdg = self.s_data[b"pdgid"]
        for i in range(pdg.size()): 
            if pdg[i] == 0: continue
            p = pdgid(int(pdg[i]))
            if p not in self.spec_pdgid_kinematics: self.spec_pdgid_kinematics[p] = {k : [] for k in kx}
            for k in kx: self.spec_pdgid_kinematics[p][k] += [self.s_data[enc(k)][i]]
        self.s_data.clear()            

        dxm = self.r_decay[b"islep"]
        for i in range(dxm.size()):
            lx = "lep" if dxm[i] else "had"
            if lx not in self.res_decay_mode: self.res_decay_mode[lx] = {k : [] for k in kx}
            for k in kx: self.res_decay_mode[lx][k] += [self.r_decay[enc(k)][i]]
            p = pdgid(int(self.r_decay[b"pdgid"][i]))
            lx = "r" + lx

            if lx not in self.top_children_dr: self.top_children_dr[lx] = []
            self.top_children_dr[lx] += [self.r_decay[b"dR"][i]]

            k = lx + "-pt"
            if k not in self.fractional: self.fractional[k] = {}
            if p not in self.fractional[k]: self.fractional[k][p] = []
            self.fractional[k][p] += [self.r_decay[b"frc.pt"][i]]

            k = lx + "-energy"
            if k not in self.fractional: self.fractional[k] = {}
            if p not in self.fractional[k]: self.fractional[k][p] = []
            self.fractional[k][p] += [self.r_decay[b"frc.e"][i]]
        self.r_decay.clear()


        dxm = self.s_decay[b"islep"]
        for i in range(dxm.size()):
            lx = "lep" if dxm[i] else "had"
            if lx not in self.spec_decay_mode: self.spec_decay_mode[lx] = {k : [] for k in kx}
            for k in kx: self.spec_decay_mode[lx][k] += [self.s_decay[enc(k)][i]]
            p = pdgid(int(self.s_decay[b"pdgid"][i]))
            lx = "s" + lx

            if lx not in self.top_children_dr: self.top_children_dr[lx] = []
            self.top_children_dr[lx] += [self.s_decay[b"dR"][i]]

            k = lx + "-pt"
            if k not in self.fractional: self.fractional[k] = {}
            if p not in self.fractional[k]: self.fractional[k][p] = []
            self.fractional[k][p] += [self.s_decay[b"frc.pt"][i]]

            k = lx + "-energy"
            if k not in self.fractional: self.fractional[k] = {}
            if p not in self.fractional[k]: self.fractional[k][p] = []
            self.fractional[k][p] += [self.s_decay[b"frc.e"][i]]
        self.s_decay.clear()

        dxm = self.top_pem[b"mass"]
        for i in range(dxm.size()):
            p  = "CT"*int(self.top_pem[b"CT"][i]) + "FT"*int(self.top_pem[b"FT"][i])
            p += "RR"*int(self.top_pem[b"RR"][i]) + "SS"*int(self.top_pem[b"SS"][i]) + "RS"*int(self.top_pem[b"RS"][i])
            if p not in self.dr_clustering: self.dr_clustering[p] = []
            self.dr_clustering[p] += [self.top_pem[b"dR"][i]]
        self.top_pem.clear()

    cdef void transform_dict_keys(self): pass

