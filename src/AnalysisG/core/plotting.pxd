# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "<plotting/plotting.h>":
    cdef cppclass plotting:
        plotting() except +
        string build_path() except +
        float get_max(string) except +
        float get_min(string) except +
        float sum_of_weights() except +

        string filename
        string extension
        string output_path

        string title
        string ytitle
        string xtitle
        string style
        string histfill
        string overflow

        string color
        vector[string] colors

        bool stack
        bool density
        bool x_logarithmic
        bool y_logarithmic

        float line_width
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

        map[string, float] x_labels
        map[string, float] y_labels

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
    cdef void __compile__(self)
    cdef void __resetplt__(self)
    cdef void __figure__(self)

cdef class TH1F(BasePlotting):
    cdef public bool ApplyScaling
    cdef public list Histograms
    cdef public TH1F Histogram

    cdef float scale_f(self)
    cdef dict factory(self)
    cdef void __error__(self, vector[float] xarr, vector[float] up, vector[float] low)

    cdef __build__(self)
    cdef void __compile__(self)
    cdef void __get_error_seg__(self, plot)

cdef class TH2F(BasePlotting):
    cdef public bool ApplyScaling
    cdef __build__(self)
    cdef void __compile__(self)

cdef class TLine(BasePlotting):
    pass
