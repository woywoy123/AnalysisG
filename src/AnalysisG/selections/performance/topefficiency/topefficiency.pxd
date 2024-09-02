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

        map[string, map[string, vector[int]]]   p_ntops
        map[string, map[string, vector[int]]]   t_ntops

        map[string, map[string, map[string, vector[float]]]] p_decaymode_topmass
        map[string, map[string, map[string, vector[float]]]] t_decaymode_topmass

        map[string, map[string, map[string, vector[float]]]] p_decaymode_zmass
        map[string, map[string, map[string, vector[float]]]] t_decaymode_zmass

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

    cdef public dict p_ntops
    cdef public dict t_ntops

    cdef public dict p_decaymode_topmass
    cdef public dict t_decaymode_topmass

    cdef public dict p_decaymode_zmass
    cdef public dict t_decaymode_zmass

    cdef public list truth_res_edge
    cdef public list truth_top_edge

    cdef public list truth_ntops
    cdef public list truth_signal

    cdef public list pred_res_edge_score
    cdef public list pred_top_edge_score

    cdef public list pred_ntops_score
    cdef public list pred_signal_score
