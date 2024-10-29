# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

from AnalysisG.core.tools cimport *
from AnalysisG.core.plotting cimport plotting

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import ks_2samp

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
        com["text.usetex"] = False #self.ptr.use_latex
        com["hatch.linewidth"] = 0.1
        self.matpl.rcParams.update(com)

    cdef void __resetplt__(self):
        self.matpl.clf()
        self.matpl.cla()
        self.matpl.close("all")

        self.matpl = plt
        self.matpl.rcdefaults()
        self.__figure__()

    cdef dict __compile__(self, bool raw = False): return {}

    def __dealloc__(self): del self.ptr
    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue

    def __reduce__(self):
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        cdef dict out = {}
        out["data"] = {i : getattr(self, i) for i in keys if not callable(getattr(self, i))}
        return self.__class__, (out,)

    def __add__(self, BasePlotting other):
        self.ptr.x_data.insert( self.ptr.x_data.end() , other.ptr.x_data.begin() , other.ptr.x_data.end())
        self.ptr.y_data.insert( self.ptr.y_data.end() , other.ptr.y_data.begin() , other.ptr.y_data.end())
        self.ptr.weights.insert(self.ptr.weights.end(), other.ptr.weights.begin(), other.ptr.weights.end())
        return self

    def __radd__(self, other):
        if isinstance(other, BasePlotting): return self.__add__(other)
        cdef BasePlotting s = self.__class__()
        return s.__add__(self)

    @property
    def Hatch(self): return env(self.ptr.hatch)
    @Hatch.setter
    def Hatch(self, str val): self.ptr.hatch = enc(val)

    @property
    def DPI(self): return self.ptr.dpi
    @DPI.setter
    def DPI(self, int val): self.ptr.dpi = val

    @property
    def Style(self): return env(self.ptr.style)
    @Style.setter
    def Style(self, str val):
        if val == "ATLAS":
            self.matpl.style.use(hep.style.ATLAS)
            self.xScaling = 20*6.4
            self.yScaling = 20*4.8
            self.DPI = 800
        self.ptr.style = enc(val)

    @property
    def ErrorBars(self): return self.ptr.errors
    @ErrorBars.setter
    def ErrorBars(self, bool val): self.ptr.errors = val

    @property
    def Filename(self): return env(self.ptr.filename)
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
    def LineStyle(self): return env(self.ptr.linestyle)
    @LineStyle.setter
    def LineStyle(self, str val): self.ptr.linestyle = enc(val)

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

    @property
    def Overflow(self): return env(self.ptr.overflow)
    @Overflow.setter
    def Overflow(self, val):
        if isinstance(val, int): self.ptr.overflow = enc("sum" if val else "none")
        else: self.ptr.overflow = env(val)

    @property
    def Color(self):
        if len(self.ptr.color): return env(self.ptr.color)
        #self.Color = next(self._ax._get_lines.prop_cycler)["color"]
        return ""

    @Color.setter
    def Color(self, str val): self.ptr.color = enc(val)

    @property
    def Colors(self): return [env(i) for i in self.ptr.colors]
    @Colors.setter
    def Colors(self, vals):
        if not isinstance(vals, list): return
        self.ptr.colors = enc_list(vals)

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

        cdef dict com = {}
        com["font.size"] = self.ptr.font_size
        com["axes.labelsize"] = self.ptr.axis_size
        com["legend.fontsize"] = self.ptr.legend_size
        com["figure.titlesize"] = self.ptr.title_size
        com["text.usetex"] = self.ptr.use_latex
        com["hatch.linewidth"] = 0.1
        self.matpl.rcParams.update(com)

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
        print("Finished Plotting: " + env(out))

cdef class TH1F(BasePlotting):

    def __init__(self, inpt = None):
        self.Histograms = []
        self.Histogram = None
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue
            except: pass

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
    def Alpha(self, float v): self.ptr.alpha = v

    @property
    def Density(self): return self.ptr.density
    @Density.setter
    def Density(self, bool val): self.ptr.density = val

    @property
    def counts(self):
        try: return sum(self.__compile__(True)["H"]).counts()
        except ValueError: return []

    @property
    def xLabels(self): return as_basic_dict(&self.ptr.x_labels)
    @xLabels.setter
    def xLabels(self, dict val): as_map(val, &self.ptr.x_labels)

    @property
    def Weights(self): return self.ptr.weights
    @Weights.setter
    def Weights(self, list val): self.ptr.weights = <vector[float]>(val)

    def KStest(self, TH1F hist, bool normalize = True):
        hist_min = hist.xMin
        hist_max = hist.xMax
        hist_bin = hist.xBins

        hist.xMin = self.xMin
        hist.xMax = self.xMax
        hist.xBins = self.xBins

        h1 = sum(hist.__compile__(True)["H"])
        h2 = sum(self.__compile__(True)["H"])

        hist.xMin  = hist_min
        hist.xMax  = hist_max
        hist.xBins = hist_bin

        v1, v2 = h1.values(), h2.values()
        w1, w2 = h1.axes[0].widths, h2.axes[0].widths
        if normalize: v1, v2 = (v1*w1)/((v1*w1).sum()), (v2*w2)/((v2*w2).sum())
        return ks_2samp(v2, v1)

    cdef void __error__(self, vector[float] xarr, vector[float] up, vector[float] low):
        self.matpl.fill_between(
                xarr, low, up,
                facecolor = "k",
                hatch = "/////",
                step  = "mid",
                alpha = 0.2
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
        return self.CrossSection * self.IntegratedLuminosity

    cdef dict factory(self):
        cdef dict histpl = {}
        histpl["histtype"] = self.HistFill
        histpl["yerr"] = self.ErrorBars
        histpl["stack"] = self.Stacked
        histpl["hatch"] = [] if not len(self.Hatch) else [self.Hatch]
        histpl["linewidth"] = self.LineWidth
        histpl["edgecolor"] = "black"
        histpl["alpha"] = self.Alpha
        histpl["binticks"] = True
        histpl["edges"] = True
        histpl["density"] = self.Density
        histpl["flow"] = self.Overflow
        histpl["label"] = []
        histpl["H"] = []
        histpl["color"] = []
        if len(self.Color): histpl["color"] += [self.Color]
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

    cdef dict __compile__(self, bool raw = False):
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

        y_max, y_min = None, None
        if self.set_ymin: y_min = self.ptr.y_min
        if self.set_ymax: y_max = self.ptr.y_max

        cdef TH1F h
        cdef dict histpl = self.factory()
        if self.Histogram is not None:
            if not len(labels): self.Histogram.xMin  = self.xMin
            if not len(labels): self.Histogram.xMax  = self.xMax
            if not len(labels): self.Histogram.xBins = self.xBins
            #histpl["H"] += [self.Histogram.__build__()]
            #histpl["label"] += [self.Histogram.Title]
            #if len(self.Histogram.Color): histpl["color"] += [self.Histogram.Color]
            #if len(self.Histogram.Hatch): histpl["hatch"] += [self.Histogram.Hatch]

        if len(self.xData) or len(labels):
            histpl = self.factory()
            histpl["label"] = None
            histpl["H"] = [self.__build__()]
            if raw: return histpl
            hep.histplot(**histpl)

        for h in self.Histograms:
            if not len(labels): h.xMin  = self.xMin
            if not len(labels): h.xMax  = self.xMax
            if not len(labels): h.xBins = self.xBins
            histpl["H"] += [h.__build__()]
            histpl["label"]  += [h.Title]
            if len(h.Color): histpl["color"] += [h.Color]
            if len(h.Hatch): histpl["hatch"] += [h.Hatch]

        if raw: return histpl
        if not len(histpl["H"]): return {}
        if self.ErrorBars and self.Histogram is not None:
            l = list(histpl["label"])
            lg = list(histpl["H"])
            del histpl["edgecolor"]
            histpl["histtype"] = "step"
            histpl["H"] = [self.Histogram.__build__()]
            histpl["label"] = [self.Histogram.Title + " (Uncertainty)"]
            error = hep.histplot(**histpl)

            histpl["histtype"] = "fill"
            histpl["edgecolor"] = "black"
            histpl["label"] = l
            histpl["H"] = lg
            hep.histplot(**histpl)
            self.__get_error_seg__(error[0])

        elif self.Histogram is not None and len(self.Histograms) and self.Stacked:
            cpy = dict(histpl)
            try: cpy["color"] = histpl["color"].pop(0)
            except IndexError: pass

            cpy["H"] = [histpl["H"].pop(0)]
            cpy["label"] = [histpl["label"].pop(0)]
            cpy["stack"] = False
            cpy["hatch"] = "///" if not len(self.Histogram.Hatch) else [histpl["hatch"].pop(0)]

            hep.histplot(**cpy)
            hep.histplot(**histpl)

        else: hep.histplot(**histpl)

        if not len(labels): 
            self.matpl.xlim(_min, _max)
            self.matpl.ylim(y_min, y_max)
        self.matpl.legend(loc = "upper right")
        return {}

cdef class TH2F(BasePlotting):
    def __cinit__(self): pass

    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue

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

    @property
    def Weights(self): return self.ptr.weights
    @Weights.setter
    def Weights(self, list val): self.ptr.weights = <vector[float]>(val)

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
            bh.axis.Regular(self.ptr.y_bins, y_min, y_max),
            storage = bh.storage.Weight()
        )

        if not self.ptr.weights.size(): h.fill(self.ptr.x_data, self.ptr.y_data)
        else: h.fill(self.ptr.x_data, self.ptr.y_data, weight = self.ptr.weights)
        return h

    cdef dict __compile__(self, bool raw = False):
        cdef dict histpl = {}
        histpl["H"] = self.__build__()
        if len(self.Color): histpl["cmap"] = self.Color
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
        return {}

cdef class TLine(BasePlotting):
    def __cinit__(self): pass

    def __init__(self, inpt = None):
        self.Lines = []
        self.LineWidth = 0.5
        self.Marker = ""
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue

    @property
    def xData(self): return self.ptr.x_data;

    @xData.setter
    def xData(self, list val): self.ptr.x_data = <vector[float]>(val)

    @property
    def yData(self): return self.ptr.y_data;

    @yData.setter
    def yData(self, list val): self.ptr.y_data = <vector[float]>(val)

    @property
    def Marker(self): return env(self.ptr.marker)

    @Marker.setter
    def Marker(self, str v): self.ptr.marker = enc(v)

    @property
    def yDataUp(self): return self.ptr.y_error_up

    @yDataUp.setter
    def yDataUp(self, val): self.ptr.y_error_up = <vector[float]>(val)

    @property
    def yDataDown(self): return self.ptr.y_error_down

    @yDataDown.setter
    def yDataDown(self, val): self.ptr.y_error_down = <vector[float]>(val)

    @property
    def LineWidth(self): return self.ptr.line_width

    @LineWidth.setter
    def LineWidth(self, float val): self.ptr.line_width = val

    cdef void factory(self):
        cdef dict coms = {}
        coms["linestyle"] = self.LineStyle
        if len(self.Color): coms["color"]
        coms["marker"] = self.Marker
        coms["linewidth"] = self.LineWidth
        coms["label"] = self.Title
        if not len(self.xData): return
        elif self.ErrorBars:
            self.ptr.build_error()
            coms["yerr"] = [self.yDataDown, self.yDataUp]
            coms["capsize"] = 3
            self.matpl.errorbar(self.xData, self.yData, **coms)
        else: self.matpl.plot(self.xData, self.yData, **coms)

    cdef dict __compile__(self, bool raw = False):

        cdef float x_max, x_min
        if self.set_xmin: x_min = self.ptr.x_min
        elif not self.ptr.x_data.size(): x_min = -1
        else: x_min = self.ptr.get_min(b"x")

        if self.set_xmax: x_max = self.ptr.x_max
        elif not self.ptr.x_data.size(): x_max = -1
        else: x_max = self.ptr.get_max(b"x")

        cdef float y_max, y_min
        if self.set_ymin: y_min = self.ptr.y_min
        elif not self.ptr.y_data.size(): y_min = -1
        else: y_min = self.ptr.get_min(b"y")

        if self.set_ymax: y_max = self.ptr.y_max
        elif not self.ptr.y_data.size(): y_max = -1
        else: y_max = self.ptr.get_max(b"y")

        if x_max > x_min: self.matpl.xlim(x_min, x_max)
        if y_max > y_min: self.matpl.ylim(y_min, y_max)
        self.matpl.title(self.Title)

        if not len(self.Lines): return {}
        cdef TLine i
        for i in self.Lines: i.factory()
        self._ax.tick_params(axis = "x", which = "minor", bottom = False)
        self.matpl.legend(loc = "upper right")
        self.factory()
        return {}
