from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../plotting/plotting.h":
    struct paint_t:
        float n_events
        bool atlas_style
        bool root_style
        bool mpl_style
        bool autoscale
        bool latex

        float xscaling
        float yscaling

        float font_size
        float label_size
        float title_size
        float legend_size
        string legend_loc

        string color
        vector[string] colors

        int atlas_loc
        float atlas_lumi
        float atlas_com
        float atlas_year
        bool atlas_data

    struct figure_t:
        string title
        bool histogram
        bool line

    struct io_t:
        string filename
        string outputdir
        int dpi

    struct axis_t:
        string title
        string dim

        int bins
        float start
        float end

        bool underflow
        bool overflow

    struct abstract_plot:
        paint_t  cosmetic
        figure_t figure
        io_t     file
        axis_t   x
        axis_t   y

