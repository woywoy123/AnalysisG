# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport as_dict, as_list
from AnalysisG.core.selection_template cimport *
from AnalysisG.selections.performance.topefficiency.topefficiency cimport *

cdef class TopEfficiency(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new topefficiency()
        self.tt = <topefficiency*>self.ptr

    def __init__(self, inpt = None):
        if inpt is None: return
        self.n_tops_predictions           = inpt["n_tops_predictions"]
        self.predicted_topmass            = inpt["predicted_topmass"]
        self.predicted_zprime_mass        = inpt["predicted_zprime_mass"]
        self.pred_res_edge_score          = inpt["pred_res_edge_score"]
        self.pred_top_edge_score          = inpt["pred_top_edge_score"]
        self.pred_ntops_score             = inpt["pred_ntops_score"]
        self.pred_signal_score            = inpt["pred_signal_score"]

        self.truth_topmass                = inpt["truth_topmass"]
        self.truth_zprime_mass            = inpt["truth_zprime_mass"]
        self.truth_res_edge               = inpt["truth_res_edge"]
        self.truth_top_edge               = inpt["truth_top_edge"]
        self.truth_ntops                  = inpt["truth_ntops"]
        self.truth_signal                 = inpt["truth_signal"]
        self.n_tops_real                  = inpt["n_tops_real"]

    def __dealloc__(self): del self.tt

    def __reduce__(self):
        cdef dict out = {
            "predicted_topmass"            : self.predicted_topmass,
            "truth_topmass"                : self.truth_topmass,
            "predicted_zprime_mass"        : self.predicted_zprime_mass,
            "truth_zprime_mass"            : self.truth_zprime_mass,
            "n_tops_predictions"           : self.n_tops_predictions,
            "n_tops_real"                  : self.n_tops_real,
            "truth_res_edge"               : self.truth_res_edge,
            "truth_top_edge"               : self.truth_top_edge,
            "truth_ntops"                  : self.truth_ntops,
            "truth_signal"                 : self.truth_signal,
            "pred_res_edge_score"          : self.pred_res_edge_score,
            "pred_top_edge_score"          : self.pred_top_edge_score,
            "pred_ntops_score"             : self.pred_ntops_score,
            "pred_signal_score"            : self.pred_signal_score
        }
        return TopEfficiency, (out,)

    cdef void transform_dict_keys(self):
        self.truth_topmass            = as_dict(&self.tt.truth_topmass)
        self.truth_zprime_mass        = as_dict(&self.tt.truth_zprime_mass)

        self.predicted_topmass        = as_dict(&self.tt.predicted_topmass)
        self.predicted_zprime_mass    = as_dict(&self.tt.predicted_zprime_mass)

        self.n_tops_predictions  = as_dict(&self.tt.n_tops_predictions)
        self.n_tops_real         = as_dict(&self.tt.n_tops_real)

        self.truth_res_edge      = self.tt.truth_res_edge
        self.truth_top_edge      = self.tt.truth_top_edge
        self.truth_ntops         = self.tt.truth_ntops
        self.truth_signal        = self.tt.truth_signal
        self.pred_res_edge_score = self.tt.pred_res_edge_score
        self.pred_top_edge_score = self.tt.pred_top_edge_score
        self.pred_ntops_score    = self.tt.pred_ntops_score
        self.pred_signal_score   = self.tt.pred_signal_score
