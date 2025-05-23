# distutils: language = c++
# cython: language_level = 3

from AnalysisG.core.notification cimport notification
from AnalysisG.core.tools cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp cimport bool, float
from cython.operator cimport dereference as deref

cdef extern from "<plotting/plotting.h>" nogil:

    cdef cppclass plotting(notification, tools):
        plotting() except+ nogil
        string build_path() except+ nogil
        float get_max(string) except+ nogil
        float get_min(string) except+ nogil
        float sum_of_weights() except+ nogil
        float  min(vector[float]*) except+ nogil
        double min(vector[double]*) except+ nogil
        void build_error() except+ nogil

        string filename
        string extension
        string output_path

        string title
        string ytitle
        string xtitle
        string style
        string histfill
        string overflow
        string marker
        string linestyle
        string hatch

        string color
        vector[string] colors

        bool stack
        bool density
        bool counts
        bool x_logarithmic
        bool y_logarithmic

        float line_width
        float cap_size
        float alpha
        float x_step
        float y_step

        float x_min
        float y_min
        float x_max
        float y_max

        int x_bins
        int y_bins
        bool errors

        vector[float] x_data
        vector[float] y_data

        vector[float] y_error_up
        vector[float] y_error_down

        unordered_map[string, float] x_labels
        unordered_map[string, float] y_labels

        vector[float] variable_x_bins
        vector[float] variable_y_bins

        vector[float] weights
        float cross_section
        float integrated_luminosity

        float font_size
        float axis_size
        float legend_size
        float title_size
        bool use_latex

        int dpi
        float xscaling
        float yscaling
        bool auto_scale

cdef class BasePlotting:
    cdef plotting* ptr
    cdef matpl
    cdef _ax
    cdef _fig

    cdef bool set_xmin
    cdef bool set_xmax

    cdef bool set_ymin
    cdef bool set_ymax

    cdef list __ticks__(self, float s, float e, float st)
    cdef dict __compile__(self, bool raw = *)
    cdef void __resetplt__(self)
    cdef void __figure__(self, dict com = *)

cdef class TH1F(BasePlotting):
    cdef public bool ApplyScaling
    cdef public list Histograms
    cdef public TH1F Histogram
    cdef fx

    cdef __error__(self, vector[float] xarr, vector[float] up, vector[float] low, str label = *, str color = *)
    cdef __get_error_seg__(self, plot, str label = *, str color = *)

    cdef float scale_f(self)
    cdef dict factory(self)
    cdef __build__(self)
    cdef dict __compile__(self, bool raw = *)

cdef class TH2F(BasePlotting):
    cdef public bool ApplyScaling
    cdef __build__(self)
    cdef dict __compile__(self, bool raw = *)

cdef class TLine(BasePlotting):
    cdef public list Lines
    cdef public bool ErrorShade
    cdef public bool ApplyScaling
    cdef __error__(self, vector[float] xarr, vector[float] up, vector[float] low, str label = *, str color = *)

    cdef void factory(self)
    cdef dict __compile__(self, bool raw = *)


