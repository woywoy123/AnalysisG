/**
 * @file plotting.pxd
 * @brief Provides type definitions and class declarations for plotting utilities in the AnalysisG framework.
 *
 * This file defines the structure and behavior of plotting-related classes, including base plotting,
 * histogram plotting, and ROC curve plotting.
 */

# distutils: language = c++
# cython: language_level = 3

from .notification cimport notification
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "<plotting/plotting.h>" nogil:
    cdef cppclass plotting(notification):
        plotting() except+ nogil
        string build_path() except+ nogil
        float get_max(string) except+ nogil
        float get_min(string) except+ nogil
        float sum_of_weights() except+ nogil
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
        vector[vector[float]] roc_data

        vector[float] y_error_up
        vector[float] y_error_down

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

/**
 * @class BasePlotting
 * @brief Base class for plotting utilities.
 *
 * This class provides common properties and methods for configuring and rendering plots.
 */
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

    cdef public float xScaling ///< Scaling factor for the x-axis.
    cdef public float yScaling ///< Scaling factor for the y-axis.
    cdef public bool AutoScaling ///< Flag to enable or disable automatic scaling.
    cdef public str Title ///< Title of the plot.
    cdef public str xTitle ///< Label for the x-axis.
    cdef public str yTitle ///< Label for the y-axis.
    cdef public bool xLogarithmic ///< Flag to enable or disable logarithmic scaling on the x-axis.
    cdef public bool yLogarithmic ///< Flag to enable or disable logarithmic scaling on the y-axis.
    cdef public list Colors ///< List of colors used in the plot.

    cdef void factory(self) ///< Factory method for initializing plot components.
    cdef dict __compile__(self, bool raw = *) ///< Compiles the plot configuration into a dictionary.

cdef class TH1F(BasePlotting):
    cdef public bool ApplyScaling
    cdef public list Histograms
    cdef public TH1F Histogram
    cdef fx

    cdef float scale_f(self)
    cdef dict factory(self)
    cdef void __error__(self, vector[float] xarr, vector[float] up, vector[float] low)

    cdef __build__(self)
    cdef dict __compile__(self, bool raw = *)
    cdef void __get_error_seg__(self, plot)

cdef class TH2F(BasePlotting):
    cdef public bool ApplyScaling
    cdef __build__(self)
    cdef dict __compile__(self, bool raw = *)

cdef class TLine(BasePlotting):
    cdef public list Lines
    cdef public bool ApplyScaling

    cdef void factory(self)
    cdef dict __compile__(self, bool raw = *)

/**
 * @class ROC
 * @brief Class for generating Receiver Operating Characteristic (ROC) curves.
 *
 * This class extends TLine and provides methods for computing and visualizing ROC curves.
 */
cdef class ROC(TLine):
    cdef bool inits ///< Initialization flag for ROC computation.
    cdef int num_cls ///< Number of classes for multi-class ROC.
    cdef public bool Binary ///< Flag to indicate binary classification.
    cdef public dict auc ///< Dictionary to store Area Under Curve (AUC) values.

    cdef void factory(self) ///< Factory method for initializing ROC components.
    cdef dict __compile__(self, bool raw = *) ///< Compiles the ROC configuration into a dictionary.

