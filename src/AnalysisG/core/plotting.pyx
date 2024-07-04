# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.string cimport string

from AnalysisG.core.tools cimport *
from AnalysisG.core.plotting cimport plotting

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

cdef class BasePlotting:
    def __cinit__(self):
        self.ptr = new plotting()
        self.matpl = plt
        self.__resetplt__()

        self.ApplyScaling = False
        self.set_xmin = False
        self.set_xmax = False
        self.set_ymin = False
        self.set_ymax = False

    cdef void __figure__(self):
        cdef dict com = {}
        com["figsize"] = (self.ptr.xscaling, self.ptr.yscaling)
        self._fig, self._ax = self.matpl.subplots(**com)
        self._ax.set_autoscale_on(self.ptr.auto_scale)

        com = {}
        com["font.size"] = self.ptr.font_size
        com["axes.labelsize"] = self.ptr.axis_size
        com["legend.fontsize"] = self.ptr.legend_size
        com["figure.titlesize"] = self.ptr.title_size
        com["text.usetex"] = self.ptr.use_latex
        com["hatch.linewidth"] = 0.1
        self.matpl.rcParams.update(com)

    cdef void __resetplt__(self):
        self.matpl.clf()
        self.matpl.cla()
        self.matpl.close("all")

        self.matpl = plt
        self.matpl.rcdefaults()
        self.__figure__()

    cdef void __compile__(self): pass

    def __dealloc__(self): del self.ptr
    def __init__(self): pass

    @property
    def ErrorBars(self): return self.ptr.errors
    @ErrorBars.setter
    def ErrorBars(self, bool val): self.ptr.errors = val

    @property
    def Filename(self): return self.ptr.filename
    @Filename.setter
    def Filename(self, str val): self.ptr.filename = enc(val)

    @property
    def OutputDirectory(self): return self.ptr.output_path
    @OutputDirectory.setter
    def OutputDirectory(self, str val): self.ptr.output_path = enc(val)

    @property
    def FontSize(self): return self.ptr.font_size
    @FontSize.setter
    def FontSize(self, float val): self.ptr.font_size = val

    @property
    def AxisSize(self): return self.ptr.axis_size
    @AxisSize.setter
    def AxisSize(self, float val): self.ptr.axis_size = val

    @property
    def LegendSize(self): return self.ptr.legend_size
    @LegendSize.setter
    def LegendSize(self, float val): self.ptr.legend_size = val

    @property
    def TitleSize(self): return self.ptr.title_size
    @TitleSize.setter
    def TitleSize(self, float val): self.ptr.title_size = val

    @property
    def UseLateX(self): return self.ptr.use_latex
    @UseLateX.setter
    def UseLateX(self, bool val): self.ptr.use_latex = val

    @property
    def xScaling(self): return self.ptr.xscaling
    @xScaling.setter
    def xScaling(self, float val): self.ptr.xscaling = val

    @property
    def yScaling(self): return self.ptr.yscaling
    @yScaling.setter
    def yScaling(self, float val): self.ptr.yscaling = val

    @property
    def AutoScaling(self): return self.ptr.auto_scale
    @AutoScaling.setter
    def AutoScaling(self, bool val): self.ptr.auto_scale = val


    @property
    def Title(self): return env(self.ptr.title)
    @Title.setter
    def Title(self, str val): self.ptr.title = enc(val)

    @property
    def xTitle(self): return env(self.ptr.xtitle)
    @xTitle.setter
    def xTitle(self, str val): self.ptr.xtitle = enc(val)

    @property
    def yTitle(self): return env(self.ptr.ytitle)
    @yTitle.setter
    def yTitle(self, str val): self.ptr.ytitle = enc(val)

    @property
    def xMin(self): return self.ptr.x_min
    @xMin.setter
    def xMin(self, float val):
        self.set_xmin = True
        self.ptr.x_min = val

    @property
    def yMin(self): return self.ptr.y_min
    @yMin.setter
    def yMin(self, float val):
        self.set_ymin = True
        self.ptr.y_min = val

    @property
    def xMax(self): return self.ptr.x_max
    @xMax.setter
    def xMax(self, float val):
        self.set_xmax = True
        self.ptr.x_max = val

    @property
    def yMax(self): return self.ptr.y_max
    @yMax.setter
    def yMax(self, float val):
        self.set_ymax = True
        self.ptr.y_max = val



    def SaveFigure(self):
        cdef string out = self.ptr.build_path()

        self.__compile__()
        self._ax.set_title(self.Title)
        self.matpl.xlabel(self.xTitle, size = self.AxisSize)
        self.matpl.ylabel(self.yTitle, size = self.AxisSize)

        self._ax.tick_params(axis = "x", which = "minor", bottom = False)

        self.matpl.tight_layout()
        self.matpl.savefig(env(out), dpi = self.ptr.dpi)
        self.matpl.close("all")



cdef class TH1F(BasePlotting):

    def __init__(self): pass

    cdef void __error__(self, vector[float] xarr, vector[float] up, vector[float] low):
        self.matpl.fill_between(xarr, low, up, facecolor = "k", alpha = 0.1, hatch = "/////", step = "mid")

    cdef void __get_error_seg__(self, plot):
        error = plot.errorbar.lines[2][0]

        cdef list k
        cdef vector[float] x_arr = []
        cdef vector[float] y_err_up = []
        cdef vector[float] y_err_lo = []

        for i in error.get_segments():
            k = i.tolist()
            x_arr.push_back(k[0][0])
            y_err_lo.push_back(k[0][1])
            y_err_up.push_back(k[1][1])
        self.__error__(x_arr, y_err_lo, y_err_up)

    @property
    def xData(self): return self.ptr.x_data;
    @xData.setter
    def xData(self, list val): self.ptr.x_data = <vector[float]>(val)

    @property
    def xBins(self): return self.ptr.x_bins
    @xBins.setter
    def xBins(self, int val): self.ptr.x_bins = val

    @property
    def CrossSection(self): return self.ptr.cross_section
    @CrossSection.setter
    def CrossSection(self, val): self.ptr.cross_section = val

    @property
    def IntegratedLuminosity(self): return self.ptr.integrated_luminosity
    @IntegratedLuminosity.setter
    def IntegratedLuminosity(self, val): self.ptr.integrated_luminosity

    @property
    def HistFill(self): return env(self.ptr.histfill)
    @HistFill.setter
    def HistFill(self, str val): self.ptr.histfill = enc(val)

    cdef void __compile__(self):
        cdef float _max, _min
        if not self.set_xmin: _min = self.ptr.get_min(b"x")
        else: _min = self.ptr.x_min

        if not self.set_xmax: _max = self.ptr.get_max(b"x")
        else: _max = self.ptr.x_max

        h = bh.Histogram(bh.axis.Regular(self.ptr.x_bins, _min, _max), storage = bh.storage.Weight())
        if self.ptr.weights.size(): h.fill(self.ptr.x_data, weight = self.ptr.weights)
        else: h.fill(self.ptr.x_data)
        if self.ApplyScaling: h *= self.CrossSection*self.IntegratedLuminosity / self.ptr.sum_of_weights()

        cdef dict histpl = {}
        histpl["H"] = [h]
        histpl["histtype"] = self.HistFill
        #histpl["yerr"] = self.ErrorBars
        #histpl["linewidth"] = self.LineWidth
        #histpl["stack"] = self.Stack
        #histpl["edgecolor"] = "black"
        #histpl["alpha"] = self.Alpha
        histpl["binticks"] = True
        #histpl["density"] = self._norm
        #histpl["flow"] = "sum" if self.OverFlow or self.UnderFlow else None

        error = hep.histplot(**histpl)
        try: self.__get_error_seg__(error[0])
        except: pass

        self.matpl.xlim(_min, _max)

cdef class TH2F(BasePlotting):
    def __cinit__(self): pass

    @property
    def yBins(self): return self.ptr.y_bins

    @yBins.setter
    def yBins(self, int val): self.ptr.y_bins = val

    @property
    def xBins(self): return self.ptr.x_bins

    @xBins.setter
    def xBins(self, int val): self.ptr.x_bins = val

    @property
    def xData(self): return self.ptr.x_data;

    @xData.setter
    def xData(self, list val): self.ptr.x_data = <vector[float]>(val)

    @property
    def yData(self): return self.ptr.y_data;

    @yData.setter
    def yData(self, list val): self.ptr.y_data = <vector[float]>(val)




cdef class TLine(BasePlotting):
    def __cinit__(self): pass

