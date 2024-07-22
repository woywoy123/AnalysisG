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
    def OutputDirectory(self): return env(self.ptr.output_path)
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
    def xLogarithmic(self): return self.ptr.x_logarithmic
    @xLogarithmic.setter
    def xLogarithmic(self, bool val): self.ptr.x_logarithmic = val

    @property
    def yLogarithmic(self): return self.ptr.y_logarithmic
    @yLogarithmic.setter
    def yLogarithmic(self, bool val): self.ptr.y_logarithmic = val

    @property
    def xStep(self): return self.ptr.x_step
    @xStep.setter
    def xStep(self, float val): self.ptr.x_step = val

    @property
    def yStep(self): return self.ptr.y_step
    @yStep.setter
    def yStep(self, float val): self.ptr.y_step = val


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


    cdef list __ticks__(self, float s, float e, float st):
        cdef list tick = []
        cdef float step = s
        while step <= e:
            tick.append(step)
            step += st
        return tick

    def SaveFigure(self):
        cdef string out = self.ptr.build_path()

        self.__compile__()
        self._ax.set_title(self.Title)
        self.matpl.xlabel(self.xTitle, size = self.AxisSize)
        self.matpl.ylabel(self.yTitle, size = self.AxisSize)

        if self.xLogarithmic: self.matpl.xscale("log")
        if self.yLogarithmic: self.matpl.yscale("log")

        if self.xStep > 0: self.matpl.xticks(self.__ticks__(self.xMin, self.xMax, self.xStep))
        if self.yStep > 0: self.matpl.yticks(self.__ticks__(self.yMin, self.yMax, self.yStep))

        self.matpl.tight_layout()
        self.matpl.savefig(env(out), dpi = self.ptr.dpi)
        self.matpl.close("all")


cdef class TH1F(BasePlotting):

    def __init__(self): self.Histograms = []

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

    @property
    def Stacked(self): return self.ptr.stack
    @Stacked.setter
    def Stacked(self, bool val): self.ptr.stack = val

    @property
    def LineWidth(self): return self.ptr.line_width
    @LineWidth.setter
    def LineWidth(self, float val): self.ptr.line_width = val

    @property
    def Alpha(self): return self.ptr.alpha
    @Alpha.setter
    def Alpha(self, float v): self.ptr.alpha

    @property
    def Density(self): return self.ptr.density
    @Density.setter
    def Density(self, bool val): self.ptr.density = val

    @property
    def xLabels(self): return as_basic_dict(&self.ptr.x_labels)

    @xLabels.setter
    def xLabels(self, dict val): as_map(val, &self.ptr.x_labels)

    cdef void __error__(self, vector[float] xarr, vector[float] up, vector[float] low):
        self.matpl.fill_between(
                xarr, low, up,
                facecolor = "k",
                hatch = "/////",
                step  = "mid",
                alpha = 0.1
        )

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

    cdef float scale_f(self):
        cdef float s = self.CrossSection * self.IntegratedLuminosity
        return s/self.ptr.sum_of_weights()

    cdef dict factory(self):
        cdef dict histpl = {}
        histpl["histtype"] = self.HistFill
        histpl["yerr"] = self.ErrorBars
        histpl["stack"] = self.Stacked
        histpl["linewidth"] = self.LineWidth
        histpl["edgecolor"] = "black"
        histpl["alpha"] = self.Alpha
        histpl["binticks"] = True
        histpl["density"] = self.Density
        histpl["flow"] = "sum"
        histpl["label"] = []
        histpl["H"] = []
        return histpl

    cdef __build__(self):
        cdef dict labels = self.xLabels
        cdef float _max, _min

        if len(labels): pass
        elif self.set_xmin: _min = self.ptr.x_min
        elif not len(labels) and not len(self.xData): pass
        else: _min = self.ptr.get_min(b"x")

        if len(labels): pass
        elif self.set_xmax: _max = self.ptr.x_max
        elif not len(labels) and not len(self.xData): pass
        else: _max = self.ptr.get_max(b"x")

        h = None
        if not len(labels):
            h = bh.Histogram(bh.axis.Regular(self.ptr.x_bins, _min, _max), storage = bh.storage.Weight())
        else:
            h = bh.Histogram(bh.axis.StrCategory(list(labels)), storage = bh.storage.Weight())
            self.ptr.weights = <vector[float]>(list(labels.values()))

        if not self.ptr.weights.size(): h.fill(self.ptr.x_data)
        elif len(labels): h.fill(list(labels), weight = self.ptr.weights)
        else: h.fill(self.ptr.x_data, weight = self.ptr.weights)
        if self.ApplyScaling: h *= self.scale_f()
        return h

    cdef void __compile__(self):
        cdef dict labels = self.xLabels
        cdef float _max, _min
        if len(labels): pass
        elif self.set_xmin: _min = self.ptr.x_min
        elif not len(labels) and not len(self.xData): pass
        else: _min = self.ptr.get_min(b"x")

        if len(labels): pass
        elif self.set_xmax: _max = self.ptr.x_max
        elif not len(labels) and not len(self.xData): pass
        else: _max = self.ptr.get_max(b"x")

        cdef TH1F h
        cdef dict histpl = self.factory()
        if self.Histogram is not None:
            histpl["H"] += [self.Histogram.__build__()]
            histpl["label"] += [self.Histogram.Title]

        if len(self.xData) or len(labels):
            if not sum(list(self.ptr.weights)):
                tmp = self.factory()
                tmp["H"] = self.__build__()
                del tmp["label"]
                hep.histplot(**tmp)
            else:
                histpl["label"] += [self.Title]
                histpl["H"] += [self.__build__()]

        for h in self.Histograms:
            if not len(labels): h.xMin  = self.xMin
            if not len(labels): h.xMax  = self.xMax
            if not len(labels): h.xBins = self.xBins
            histpl["H"] += [h.__build__()]
            histpl["label"] += [h.Title]

        if not len(histpl["H"]): return
        error = hep.histplot(**histpl)
        try: self.__get_error_seg__(error[0])
        except: pass

        if not len(labels): self.matpl.xlim(_min, _max)
        self.matpl.legend(loc = "upper right")

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

    cdef __build__(self):
        cdef float x_max, x_min
        if self.set_xmin: x_min = self.ptr.x_min
        else: x_min = self.ptr.get_min(b"x")

        if self.set_xmax: x_max = self.ptr.x_max
        else: x_max = self.ptr.get_max(b"x")

        cdef float y_max, y_min
        if self.set_ymin: y_min = self.ptr.y_min
        else: y_min = self.ptr.get_min(b"y")

        if self.set_ymax: y_max = self.ptr.y_max
        else: y_max = self.ptr.get_max(b"y")

        h = bh.Histogram(
            bh.axis.Regular(self.ptr.x_bins, x_min, x_max),
            bh.axis.Regular(self.ptr.y_bins, y_min, y_max)
        )

        h.fill(self.ptr.x_data, self.ptr.y_data)
        return h

    cdef void __compile__(self):
        cdef dict histpl = {}
        histpl["H"] = self.__build__()
        error = hep.hist2dplot(**histpl)

        cdef float x_max, x_min
        if self.set_xmin: x_min = self.ptr.x_min
        else: x_min = self.ptr.get_min(b"x")

        if self.set_xmax: x_max = self.ptr.x_max
        else: x_max = self.ptr.get_max(b"x")

        cdef float y_max, y_min
        if self.set_ymin: y_min = self.ptr.y_min
        else: y_min = self.ptr.get_min(b"y")

        if self.set_ymax: y_max = self.ptr.y_max
        else: y_max = self.ptr.get_max(b"y")
        self.matpl.xlim(x_min, x_max)
        self.matpl.ylim(y_min, y_max)

cdef class TLine(BasePlotting):
    def __cinit__(self): pass

