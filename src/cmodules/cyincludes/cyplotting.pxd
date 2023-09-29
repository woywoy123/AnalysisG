from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

from cytypes cimport metric_t
from cymetrics cimport roc_t
from cyplotstruct cimport *


cdef extern from "../plotting/plotting.h":
    cdef cppclass CyPlotting nogil:
        CyPlotting() except +
        paint_t  painter_params
        io_t     file_params
        figure_t figure_params

    cdef cppclass CyMetric nogil:
        CyMetric() except +

        vector[roc_t*] FetchROC() except +
        void AddMetric(map[string, metric_t]* values, string mode) except +
        void BuildPlots(map[string, abstract_plot]* output) except +

        int current_epoch
        string outpath
        report_t report

