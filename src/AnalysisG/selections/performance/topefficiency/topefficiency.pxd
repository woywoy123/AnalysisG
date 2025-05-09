# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "topefficiency.h":
    cdef cppclass topefficiency(selection_template):
        topefficiency() except +

        map[string, map[string, vector[float]]] p_topmass
        map[string, map[string, vector[float]]] t_topmass

        map[string, map[string, vector[float]]] p_zmass
        map[string, map[string, vector[float]]] t_zmass

        map[string, map[string, vector[float]]] prob_tops
        map[string, map[string, vector[float]]] prob_zprime

        map[string, map[string, vector[float]]] t_decay_region
        map[string, map[string, vector[float]]] p_decay_region

        map[string, map[float, int]] t_nodes
        map[string, map[float, int]] p_nodes

        map[string, int] n_tru_tops
        map[string, map[float, int]] n_pred_tops
        map[string, map[float, int]] n_perfect_tops

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

    cdef public dict p_topmass
    cdef public dict t_topmass

    cdef public dict p_zmass
    cdef public dict t_zmass

    cdef public dict prob_tops
    cdef public dict prob_zprime

    cdef public dict t_decay_region
    cdef public dict p_decay_region

    cdef public dict t_nodes
    cdef public dict p_nodes

    cdef public dict n_tru_tops
    cdef public dict n_pred_tops
    cdef public dict n_perfect_tops

    cdef public list truth_res_edge
    cdef public list truth_top_edge

    cdef public list truth_ntops
    cdef public list truth_signal

    cdef public list pred_res_edge_score
    cdef public list pred_top_edge_score

    cdef public list pred_ntops_score
    cdef public list pred_signal_score




