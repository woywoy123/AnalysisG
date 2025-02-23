# distuils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "regions.h":
    cdef struct regions_t:
        double variable1
        double variable2
        double weight
        bool passed

    cdef struct package_t:
        regions_t CRttbarCO2l_CO
        regions_t CRttbarCO2l_CO_2b
        regions_t CRttbarCO2l_gstr
        regions_t CRttbarCO2l_gstr_2b
        regions_t CR1b3lem
        regions_t CR1b3le
        regions_t CR1b3lm
        regions_t CRttW2l_plus
        regions_t CRttW2l_minus
        regions_t CR1bplus
        regions_t CR1bminus
        regions_t CRttW2l
        regions_t VRttZ3l
        regions_t VRttWCRSR
        regions_t SR4b
        regions_t SR2b
        regions_t SR3b
        regions_t SR2b2l
        regions_t SR2b3l4l
        regions_t SR2b4l
        regions_t SR3b2l
        regions_t SR3b3l4l
        regions_t SR3b4l
        regions_t SR4b4l
        regions_t SR

    cdef cppclass regions(selection_template):
        regions() except +
        vector[package_t] output

cdef class Regions(SelectionTemplate):
    cdef regions* tt
    cdef public list output


