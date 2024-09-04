# distutils: language=c++
# cython: language_level=3

from AnalysisG.selections.performance.topefficiency.topefficiency cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *

cdef class TopEfficiency(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topefficiency()
        self.tt = <topefficiency*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt[i])
            except KeyError: continue

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef list keys = self.__dir__()
        cdef dict out = {i : getattr(self, i) for i in keys if not i.startswith("__")}
        return TopEfficiency, (out,)

    cdef void transform_dict_keys(self):

        self.p_topmass           = as_dict_dict(&self.tt.p_topmass)
        self.t_topmass           = as_dict_dict(&self.tt.t_topmass)

        self.p_zmass             = as_dict_dict(&self.tt.p_zmass)
        self.t_zmass             = as_dict_dict(&self.tt.t_zmass)

        self.p_ntops             = as_dict_dict(&self.tt.p_ntops)
        self.t_ntops             = as_dict_dict(&self.tt.t_ntops)

        self.p_decaymode_topmass = as_dict_dict_dict(&self.tt.p_decaymode_topmass)
        self.t_decaymode_topmass = as_dict_dict_dict(&self.tt.t_decaymode_topmass)

        self.p_decaymode_zmass = as_dict_dict_dict(&self.tt.p_decaymode_zmass)
        self.t_decaymode_zmass = as_dict_dict_dict(&self.tt.t_decaymode_zmass)

        self.prob_tops       = as_dict_dict(&self.tt.prob_tops)
        self.prob_zprime     = as_dict_dict(&self.tt.prob_zprime)

        self.purity_tops     = as_dict(&self.tt.purity_tops)
        self.efficiency_tops = as_dict(&self.tt.efficiency_tops)

        self.truth_res_edge      = self.tt.truth_res_edge
        self.truth_top_edge      = self.tt.truth_top_edge

        self.truth_ntops         = self.tt.truth_ntops
        self.truth_signal        = self.tt.truth_signal

        self.pred_res_edge_score = self.tt.pred_res_edge_score
        self.pred_top_edge_score = self.tt.pred_top_edge_score

        self.pred_ntops_score    = self.tt.pred_ntops_score
        self.pred_signal_score   = self.tt.pred_signal_score
