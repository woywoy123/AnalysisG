# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "<plotting/plotting.h>":
    cdef cppclass plotting:
        plotting() except +
        string build_path() except +

        string filename
        string extension
        string output_path

        string title
        string ytitle
        string xtitle
        string style

        float x_min
        float y_min
        float x_max
        float y_max

        int x_bins
        int y_bins

        vector[float] x_data
        vector[float] y_data
        vector[float] weights

        int font_size
        int axis_size
        int legend_size
        int title_size
        bool use_latex

        float xscaling
        float yscaling
        bool auto_scale



cdef class BasePlotting:
    cdef plotting* ptr
    cdef void __compile__(self)
    cdef void __resetplt__(self)
    cdef void __figure__(self)

cdef class TH1F(BasePlotting):
    cdef void __compile__(self)

cdef class TH2F(BasePlotting):
    pass

cdef class TLine(BasePlotting):
    pass
