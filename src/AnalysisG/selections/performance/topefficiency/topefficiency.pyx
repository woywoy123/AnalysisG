# distutils: language=c++
# cython: language_level=3

from AnalysisG.selections.performance.topefficiency.topefficiency cimport *
from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *

cdef class TopEfficiency(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topefficiency()
        self.tt = <topefficiency*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):

        self.p_topmass   = as_dict_dict(&self.tt.p_topmass)
        self.t_topmass   = as_dict_dict(&self.tt.t_topmass)

        self.p_zmass     = as_dict_dict(&self.tt.p_zmass)
        self.t_zmass     = as_dict_dict(&self.tt.t_zmass)

        self.prob_tops   = as_dict_dict(&self.tt.prob_tops)
        self.prob_zprime = as_dict_dict(&self.tt.prob_zprime)

        self.ms_cut_perf_tops = as_dict_dict(&self.tt.ms_cut_perf_tops)
        self.ms_cut_reco_tops = as_dict_dict(&self.tt.ms_cut_reco_tops)
        self.ms_cut_topmass   = as_dict_dict(&self.tt.ms_cut_topmass  )

        self.n_tru_tops       = as_dict(&self.tt.n_tru_tops)

        self.kin_truth_tops   = as_dict_dict(&self.tt.kin_truth_tops  )
        self.ms_kin_perf_tops = as_dict_dict_dict(&self.tt.ms_kin_perf_tops)
        self.ms_kin_reco_tops = as_dict_dict_dict(&self.tt.ms_kin_reco_tops)

        self.truth_res_edge   = self.tt.truth_res_edge
        self.truth_top_edge   = self.tt.truth_top_edge

        self.truth_ntops      = self.tt.truth_ntops
        self.truth_signal     = self.tt.truth_signal

        self.pred_res_edge_score = self.tt.pred_res_edge_score
        self.pred_top_edge_score = self.tt.pred_top_edge_score

        self.pred_ntops_score    = self.tt.pred_ntops_score
        self.pred_signal_score   = self.tt.pred_signal_score
