# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict
from AnalysisG.core.selection_template cimport *

def res_data(zp, data):
    name, val = data
    keys = ["energy", "eta", "phi", "pt"]
    if zp.res_top_kinematics is None: zp.res_top_kinematics = {k : [] for k in keys}
    for i in keys:
        if i not in name: continue
        zp.res_top_kinematics[i] += val
        break

def spec_data(zp, data):
    name, val = data
    keys = ["energy", "eta", "phi", "pt"]
    if zp.spec_top_kinematics is None: zp.spec_top_kinematics = {k : [] for k in keys}
    for i in keys:
        if i not in name: continue
        zp.spec_top_kinematics[i] += val
        break

def mass_data(zp, data):
    name, val = data
    keys = ["RR", "RS", "SS"]
    if zp.mass_combi is None: zp.mass_combi = {k : [] for k in keys}
    for i in keys:
        if i not in name: continue
        zp.mass_combi[i] += val
        break

def dr_data(zp, data):
    name, val = data
    keys = ["RR", "RS", "SS"]
    if zp.deltaR is None: zp.deltaR = {k : [] for k in keys}
    for i in keys:
        if i not in name: continue
        zp.deltaR[i] += val
        break

cdef class TopKinematics(SelectionTemplate):
    def __cinit__(self):
        self.root_leaves = {
                "res_energy": res_data,
                "res_eta"   : res_data,
                "res_phi"   : res_data,
                "res_pt"    : res_data, 
                "spec_energy": spec_data,
                "spec_eta"   : spec_data,
                "spec_phi"   : spec_data,
                "spec_pt"    : spec_data, 
                "mass_RR"    : mass_data, 
                "mass_RS"    : mass_data, 
                "mass_SS"    : mass_data, 
                "dr_RR"      : dr_data, 
                "dr_RS"      : dr_data, 
                "dr_SS"      : dr_data
        }


        self.ptr = new topkinematics()
        self.tt = <topkinematics*>self.ptr

    def __dealloc__(self): del self.tt
