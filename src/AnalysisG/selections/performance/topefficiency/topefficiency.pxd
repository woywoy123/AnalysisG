# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topefficiency.h":
    cdef cppclass topefficiency(selection_template):
        topefficiency() except +

        map[string, vector[float]] truthchildren_pt_eta_topmass
        map[string, vector[float]] truthjets_pt_eta_topmass
        map[string, vector[float]] jets_pt_eta_topmass

        map[string, vector[float]] predicted_topmass
        map[string, vector[float]] truth_topmass

        map[string, vector[int]] n_tops_predictions
        map[string, vector[int]] n_tops_real

        vector[int] truth_res_edge
        vector[int] truth_top_edge

        vector[int] truth_ntops
        vector[int] truth_signal

        vector[vector[float]] pred_res_edge_score
        vector[vector[float]] pred_top_edge_score

        vector[vector[float]] pred_ntops_score
        vector[vector[float]] pred_signal_score


cdef class TopEfficiency(SelectionTemplate):
    cdef topefficiency* tt

    cdef public dict truthchildren_pt_eta_topmass
    cdef public dict truthjets_pt_eta_topmass
    cdef public dict jets_pt_eta_topmass

    cdef public dict predicted_topmass
    cdef public dict truth_topmass

    cdef public dict n_tops_predictions
    cdef public dict n_tops_real

    cdef public list truth_res_edge
    cdef public list truth_top_edge

    cdef public list truth_ntops
    cdef public list truth_signal

    cdef public list pred_res_edge_score
    cdef public list pred_top_edge_score

    cdef public list pred_ntops_score
    cdef public list pred_signal_score
