# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool, float

from cython.operator cimport dereference as dref
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import numpy
import random
import pathlib
import pickle


def ratio(H1, H2, axis, ylabel = "Ratio", normalize = False, yerror = False, this_hist = None):
    cdef dict out = {}
    axis.set_ylim(0, 2)
    axis.set_ylabel(ylabel, fontsize = this_hist.FontSize * (35 / len(ylabel)), loc = "center")

    if normalize:
        t1, t2 = H1.values(), H2.values()
        w1, w2 = H1.axes[0].widths, H2.axes[0].widths
        t1, t2 = (t1*w1)/((t1*w1).sum()), (t2*w2)/((t2*w2).sum())
        H1.reset()
        H2.reset()
        H1.view().value = t1
        H2.view().value = t2

    cdef float v1, v2
    cdef vector[float] h1 = H1.counts().tolist()
    cdef vector[float] h2 = H2.counts().tolist()
    cdef vector[float] s1 = H1.variances().tolist()
    cdef vector[float] s2 = H2.variances().tolist()
    for i in range(h2.size()):
        v1 = (h1[i] if h1[i] else 1)**2
        v2 = (h2[i] if h2[i] else 1)**2
        h2[i] = h1[i]/(h2[i] if h2[i] else 1)
        if not yerror: s2[i] = 0
        else: s2[i] = (h2[i]**2)*( (s1[i]/v1) + (s2[i]/v2) )

    H1.reset()
    H1.view().value    = [v1 if v1 > 0 else -1 for v1 in h2]
    H1.view().variance = [s2[i] if h2[i] > 0 else 0 for i in range(s2.size())]
    out["H"] = H1
    out["ax"] = axis
    out["color"] = "black"
    out["histtype"] = "errorbar"
    out["markersize"] = 5
    axis.axhline(1, linestyle = "--", color = "grey")
    return out

def chi2(H1, H2, axis, ylabel = "Ratio", normalize = False, yerror = False, this_hist = None):
    cdef dict out = {}
    axis.set_ylim(-1.2, 1.2)
    axis.set_ylabel("$\\frac{O_i - E_i}{|E_i|}$", fontsize = this_hist.FontSize, labelpad = 0.1)

    if normalize:
        t1, t2 = H1.values(), H2.values()
        w1, w2 = H1.axes[0].widths, H2.axes[0].widths
        t1, t2 = (t1*w1)/((t1*w1).sum()), (t2*w2)/((t2*w2).sum())
        H1.reset()
        H2.reset()
        H1.view().value = t1
        H2.view().value = t2

    cdef float _chi2 = 0
    cdef vector[float] h1 = H1.counts().tolist()
    cdef vector[float] h2 = H2.counts().tolist()
    for i in range(h2.size()):
        _chi2 += (h1[i] - h2[i])**2/(h2[i] if h2[i] else 1)
        h2[i] = (h1[i] - h2[i])/abs(h2[i] if h2[i] else 1) - 100*((h1[i] + h2[i]) == 0)
    H1.reset()
    H1.view().value = h2

    out["H"] = H1
    out["ax"] = axis
    out["color"] = "black"
    out["marker"] = this_hist.Marker
    out["histtype"] = "errorbar"
    out["markersize"] = 5
    out["add"] = "$\\left(\\chi^2 = " + "{:e}".format(_chi2) + "\\right)$"
    axis.axhline(0, linestyle = "--", color = "grey", linewidth = 1.0)
    return out

cdef class BasePlotting:
    def __cinit__(self):
        self.ptr = new plotting()
        self.matpl = plt

        self.ApplyScaling = False
        self.set_xmin = False
        self.set_xmax = False
        self.set_ymin = False
        self.set_ymax = False

    cdef void __figure__(self, dict com = {"nrows" : 1, "ncols" : 1}):
        com["figsize"] = (self.ptr.xscaling, self.ptr.yscaling)
        if "sharex" in com: com["figsize"][0]*1.5
        self._fig, self._ax = self.matpl.subplots(**com)
        try: self._ax.set_autoscale_on(self.ptr.auto_scale)
        except: self._ax[0].set_autoscale_on(self.ptr.auto_scale)

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
        cdef bool cls
        cdef dict out = {"data" : {}}
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: cls = callable(getattr(self, i))
            except: cls = False
            if cls: continue
            try: out["data"][i] = getattr(self, i)
            except: pass
        return self.__class__, (out,)

    def dump(self, str path = "", str name = ""):
        if not len(name): name = env(self.ptr.filename)
        if not len(path): path = env(self.ptr.output_path)
        pathlib.Path(path).mkdir(parents = True, exist_ok = True)
        try: pickle.dump(self, open(path + "/" + name + ".pkl", "wb"))
        except OSError: self.ptr.failure(b"Failed to save the Plotting Object")

    def load(self, str path = "", str name = ""):
        if not len(name): name = env(self.ptr.filename)
        if not len(path): path = env(self.ptr.output_path)
        try: return pickle.load(open(path + "/" + name + ".pkl", "rb"))
        except OSError: print("Failed to load the Plotting Object")
        except EOFError: print("Failed to load the Plotting Object")
        return None

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
    def Style(self, str val): self.ptr.style = enc(val)

    @property
    def CapSize(self): return self.ptr.cap_size
    @CapSize.setter
    def CapSize(self, float val): self.ptr.cap_size = val

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
    def Title(self):
        cdef str titl = env(self.ptr.title)
        if not self.ptr.counts: return titl
        if not self.ptr.x_data.size(): return titl
        return titl + " (" + str(round(float(sum(self.counts)), 3)) + ")"

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
        else: self.ptr.overflow = enc(val)

    @property
    def Color(self):
        if len(self.ptr.color): return env(self.ptr.color)
        self.Color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        return self.Color

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
        self.__resetplt__()
        if self.Style == "ATLAS": 
            self.matpl.style.use(hep.style.ATLAS)
            self.DPI = 800

        cdef string out = self.ptr.build_path()
        cdef str raw = env(out).replace(self.Filename + env(self.ptr.extension), "raw/" + self.Filename + ".pgf")
        cdef dict com = {}
        com["font.size"] = self.ptr.font_size
        com["axes.labelsize"] = self.ptr.axis_size
        com["legend.fontsize"] = self.ptr.legend_size
        com["figure.titlesize"] = self.ptr.title_size
        com["hatch.linewidth"] = self.ptr.line_width
        com["text.usetex"] = self.ptr.use_latex
        if self.ptr.use_latex:
            com["pgf.texsystem"] = "lualatex"
            com["text.latex.preamble"] = r"\usepackage{amsmath}"
            com["pgf.preamble"] = r"\usepackage{amsmath}"

        try: 
            self.matpl.rcParams.update(**com)
            self.__compile__()

            try: self._ax.set_title(self.Title, fontsize = self.ptr.title_size)
            except AttributeError: self.matpl.suptitle(self.Title, fontsize = self.ptr.font_size, y = 1.0)
            self.matpl.xlabel(self.xTitle, fontsize = self.ptr.font_size, labelpad = 0.1)

            if "<units>" not in self.yTitle: yl = self.yTitle
            else: yl = self.yTitle.replace("<units>", str(round((self.xMax - self.xMin)/self.xBins, 3)))

            try: self._ax.set_ylabel(yl, fontsize = self.ptr.font_size)
            except AttributeError: self._ax[0].set_ylabel(yl, fontsize = self.ptr.font_size)

            if self.xLogarithmic: self.matpl.xscale("log")
            if self.yLogarithmic: self.matpl.yscale("log")

            if self.ptr.variable_x_bins.size(): self.matpl.xticks(self.ptr.variable_x_bins, fontsize = self.ptr.axis_size)
            elif self.xStep > 0: self.matpl.xticks(self.__ticks__(self.xMin, self.xMax, self.xStep), fontsize = self.ptr.axis_size)
            else: self.matpl.xticks(fontsize = self.ptr.axis_size)

            if self.ptr.variable_y_bins.size(): self.matpl.yticks(self.ptr.variable_y_bins, fontsize = self.ptr.axis_size)
            elif self.yStep > 0: self.matpl.yticks(self.__ticks__(self.yMin, self.yMax, self.yStep), fontsize = self.ptr.axis_size)
            else: self.matpl.yticks(fontsize = self.ptr.axis_size)

            self.matpl.gcf().set_size_inches(self.ptr.xscaling, self.ptr.yscaling)

            com = {}
            com["dpi"] = self.ptr.dpi
            com["bbox_inches"] = "tight"
            com["pad_inches"] = 0
            com["transparent"] = True
            if self.ptr.use_latex: com["backend"] = "pgf"
            self.matpl.savefig(env(out), **com)
            if self.ptr.use_latex: self.matpl.savefig(raw, **com)
            self.matpl.close("all")
            self.ptr.success(b"Finished Plotting: " + out)
        except Exception as error:
            self.ptr.failure(b"Failed Plotting... Dumping State...")
            self.ptr.failure(enc(str(error)))
            self.dump()

cdef class TH1F(BasePlotting):

    def __cinit__(self): self.ptr.prefix = b"TH1F"

    def __init__(self, inpt = None, **kwargs):
        self.Histograms = []
        self.Histogram = None
        if len(kwargs): inpt = {"data" : dict(kwargs)}
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            if i == "yMax" or i == "yMin": continue
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue
            except: pass

    @property
    def xData(self): return self.ptr.x_data;
    @xData.setter
    def xData(self, list val): self.ptr.x_data = <vector[float]>(val)

    @property
    def xBins(self):
        if not self.ptr.variable_x_bins.size(): return self.ptr.x_bins
        else: return self.ptr.variable_x_bins

    @xBins.setter
    def xBins(self, val):
        if isinstance(val, int): self.ptr.x_bins = val
        elif isinstance(val, list): self.ptr.variable_x_bins = <vector[float]>(val)
        else: self.ptr.warning(b"Invalid Bins specified")

    @property
    def CrossSection(self): return self.ptr.cross_section
    @CrossSection.setter
    def CrossSection(self, val): self.ptr.cross_section = val

    @property
    def IntegratedLuminosity(self): return self.ptr.integrated_luminosity
    @IntegratedLuminosity.setter
    def IntegratedLuminosity(self, val): self.ptr.integrated_luminosity = val

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
        except AttributeError: return []

    @property
    def Marker(self): return env(self.ptr.marker)
    @Marker.setter
    def Marker(self, str v): self.ptr.marker = enc(v)

    @property
    def xLabels(self): return as_basic_udict(&self.ptr.x_labels)
    @xLabels.setter
    def xLabels(self, dict val): as_umap(val, &self.ptr.x_labels)

    @property
    def Weights(self): return self.ptr.weights
    @Weights.setter
    def Weights(self, list val): self.ptr.weights = <vector[float]>(val)

    @property
    def ShowCount(self): return self.ptr.counts
    @ShowCount.setter
    def ShowCount(self, bool val): self.ptr.counts = val

    def FX(self, val = None):
        if isinstance(val, str):
            if   val.lower() == "ratio": self.fx = ratio
            elif val.lower() == "chi2": self.fx = chi2
            else: self.ptr.warning(b"Input Option: " + enc(val) + " is an invalid option! (ratio, ratio_chi2)")
        elif val is None: self.fx = ratio
        else: self.fx = val

    def KStest(self, TH1F hist):
        hist_min = self.xMin
        hist_max = self.xMax
        hist_bin = self.xBins

        cdef float i
        cdef vector[float] h1 = [i for i in self.ptr.x_data if i >= hist_min and i <= hist_max]
        cdef vector[float] h2 = [i for i in hist.ptr.x_data if i >= hist_min and i <= hist_max]
        return ks_2samp(h1, h2)

    cdef __error__(self, vector[float] xarr, vector[float] up, vector[float] low, str label = "Uncertainty", str color = "k"):
        try: ax = self._ax[0]
        except: ax = self._ax
        cdef dict apl = {"step" : "post", "hatch" : "///", "alpha" : 0.15, "linewidth" : 0.0}
        if len(label): apl["label"] = label
        apl["facecolor"] = color if len(color) else None
        apl["edgecolor"] = ("k", 1.0)
        ax.fill_between(xarr, low, up, **apl)

        apl = {"hatch" : "///", "step": "post", "alpha" : 0.00}
        return self.matpl.fill_between(xarr, low, up, **apl)

    cdef __get_error_seg__(self, plot, str label = "Uncertainty", str color = "k"):
        error = plot.errorbar.lines[2][0]
        cdef list k
        cdef int ix = 0
        cdef vector[float] w_arr = []
        cdef vector[float] x_arr = []
        cdef vector[float] y_err_up = []
        cdef vector[float] y_err_lo = []
        for i in error.get_segments():
            k = i.tolist()
            x_arr.push_back(k[0][0])
            y_err_lo.push_back(k[0][1])
            if k[0][1] == 0: y_err_up.push_back(0.0)
            else: y_err_up.push_back(k[1][1])
            if ix: w_arr.push_back(abs(x_arr[ix-1] - x_arr[ix]))
            ix += 1
        ix -= 1
        for i in range(ix): x_arr[i] = x_arr[i] - w_arr[i]*0.5
        x_arr[ix] = x_arr[ix] - w_arr[ix-1]*0.5
        x_arr.push_back(x_arr[ix] + w_arr[ix-1])
        y_err_lo.push_back(y_err_lo[ix])
        y_err_up.push_back(y_err_up[ix])
        return self.__error__(x_arr, y_err_lo, y_err_up, label, color)

    cdef float scale_f(self): return self.CrossSection * self.IntegratedLuminosity

    cdef dict factory(self):
        cdef dict histpl = {}
        histpl["histtype"] = self.HistFill
        histpl["yerr"] = self.ErrorBars
        histpl["stack"] = self.Stacked
        histpl["hatch"] = [] if not len(self.Hatch) else [self.Hatch]
        histpl["linewidth"] = self.LineWidth
        histpl["edgecolor"] = "black"
        histpl["alpha"] =  []
        histpl["binticks"] = True
        histpl["edges"] = True
        histpl["density"] = self.Density
        histpl["flow"] = self.Overflow
        histpl["label"] = []
        histpl["H"] = []
        histpl["color"] = []
        if self.ptr.x_data.size(): histpl["color"] += [self.Color]
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

        if len(labels):
            ax_ = bh.axis.StrCategory(list(labels))
            self.ptr.weights = <vector[float]>(list(labels.values()))
        elif self.ptr.variable_x_bins.size(): ax_ = bh.axis.Variable(self.ptr.variable_x_bins)
        else: ax_ = bh.axis.Regular(self.ptr.x_bins, _min, _max)
        h = bh.Histogram(ax_, storage = bh.storage.Weight())

        if not self.ptr.weights.size(): h.fill(self.ptr.x_data)
        elif len(labels): h.fill(list(labels), weight = self.ptr.weights)
        else: h.fill(self.ptr.x_data, weight = self.ptr.weights)
        cdef float norm = float(sum(h.counts()))

        if self.Density: h *= 1/(norm if norm != 0 else 1)
        if self.ApplyScaling: h *= self.scale_f()
        return h

    cdef dict __compile__(self, bool raw = False):
        cdef dict labels = self.xLabels
        cdef float x_max, x_min

        if len(labels): pass
        elif self.set_xmin: x_min = self.ptr.x_min
        elif not len(labels) and not len(self.xData): pass
        elif self.ptr.variable_x_bins.size(): x_min = self.ptr.variable_x_bins.front()
        else: x_min = self.ptr.get_min(b"x")

        if len(labels): pass
        elif self.set_xmax: x_max = self.ptr.x_max
        elif not len(labels) and not len(self.xData): pass
        elif self.ptr.variable_x_bins.size(): x_max = self.ptr.variable_x_bins.back()
        else: x_max = self.ptr.get_max(b"x")

        y_max, y_min = None, None
        if self.set_ymin: y_min = self.ptr.y_min
        if self.set_ymax: y_max = self.ptr.y_max

        cdef TH1F h
        cdef dict histpl = self.factory()
        if self.Histogram is not None:
            if not len(labels): self.Histogram.xMin  = self.xMin
            if not len(labels): self.Histogram.xMax  = self.xMax
            if not len(labels): self.Histogram.xBins = self.xBins
            if self.ShowCount:  self.Histogram.ShowCount = self.ShowCount

            histpl["H"]     += [self.Histogram.__build__()]
            histpl["label"] += [self.Histogram.Title]
            histpl["color"] += [self.Histogram.Color]
            histpl["hatch"] += [self.Histogram.Hatch]
            histpl["alpha"] += [self.Histogram.Alpha]

        if len(self.xData) or len(labels):
            histpl = self.factory()
            if len(self.Histograms) and not len(self.xData): histpl["label"] = []
            else:
                histpl["label"] = None
                histpl["H"]     = [self.__build__()]
                if raw: return histpl
                hep.histplot(**histpl)

        for h in self.Histograms:
            if not len(labels): h.xMin  = self.xMin
            if not len(labels): h.xMax  = self.xMax
            if not len(labels): h.xBins = self.xBins
            if self.ShowCount:  h.ShowCount = self.ShowCount
            histpl["label"] += [h.Title]
            histpl["H"]     += [h.__build__()]
            histpl["color"] += [h.Color]
            histpl["hatch"] += [h.Hatch]
            histpl["alpha"] += [h.Alpha]

        if raw: return histpl
        if not len(histpl["H"]): return {}

        if self.ErrorBars and self.Histogram is not None and self.fx is None:
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
            self.__figure__({
                "nrows" : 2, "ncols" : 1, "sharex" : True,
                "gridspec_kw" : {
                    "height_ratios" : [4, 1], "hspace" : 0.05
                }
            })

            cpy = {}
            cpy["linewidth"] = 0
            cpy["histtype"]  = "step"
            cpy["yerr"]      = True
            cpy["ax"]        = self._ax[0]
            cpy["H"]         = sum(histpl["H"][1:])
            error            = hep.histplot(**cpy)

            cpy = {}
            cpy["color"]      = "black"
            cpy["H"]          = histpl["H"].pop(0)
            cpy["label"]      = histpl["label"].pop(0)
            cpy["alpha"]      = histpl["alpha"].pop(0)
            cpy["histtype"]   = "errorbar"
            cpy["markersize"] = 5
            cpy["linewidth"]  = 2
            cpy["stack"]      = False
            cpy["ax"]         = self._ax[0]
            hep.histplot(**cpy)

            del histpl["hatch"]
            histpl["color"].pop(0)
            histpl["ax"] = self._ax[0]

            hep.histplot(**histpl)
            self.__get_error_seg__(error[0])
            self._ax[0].legend(loc = "upper right")
            self._ax[0].set_xlim(x_min, x_max, auto = True)
            self._ax[0].set_ylim(y_min, y_max, auto = True)

            if self.fx is None: self.FX()
            cpy = self.fx(
                    sum(histpl["H"]), cpy["H"], self._ax[1],
                    "Ratio - (" + self.Histogram.Title + "/" + cpy["label"],
                    self.Density, self.ErrorBars
            )
            hep.histplot(**cpy)
            return {}

        elif self.Histogram is not None and len(self.Histograms) and not self.Stacked:
            self.__figure__({
                "nrows" : 2, "ncols" : 1, "sharex" : True,
                "gridspec_kw" : {
                    "height_ratios" : [4, 1], "hspace" : 0.03
                }
            })

            histpl["ax"] = self._ax[0]
            hep.histplot(**histpl)
            self._ax[0].legend(loc = "upper right")
            if self.fx is None: self.FX()

            if len(self.Histograms) > 1: yl = "$\\frac{\\text{" + self.Histogram.Title + "}}{H_{X}})$"
            else: yl = "$\\frac{\\text{" + self.Histogram.Title + "}}{\\text{" + self.Histograms[0].Title + "}}$"

            for i in range(1, len(histpl["H"])):
                try: cpy = self.fx(histpl["H"][0].copy(), histpl["H"][i].copy(), self._ax[1], yl, self.Density, histpl["yerr"], self.Histograms[i-1])
                except: cpy = self.fx(histpl["H"][0].copy(), histpl["H"][i].copy(), self._ax[1], self.Histograms[i-1])
                cpy["color"] = histpl["color"][i]
                if "label" not in cpy: cpy["label"] = histpl["label"][i]
                if "add" in cpy: cpy["label"] += " " + cpy["add"]; del cpy["add"]
                hep.histplot(**cpy)
            self._ax[1].legend(loc = "best")
            return {}

        elif self.Histogram is None and len(self.Histograms) and self.ErrorBars:
            hts = dict(histpl)
            del hts["hatch"]
            del hts["edgecolor"]
            del hts["label"]

            hts["alpha"] = 0.0
            hts["histtype"] = "errorbar"
            error = hep.histplot(**hts)
            hep.histplot(**histpl)

            ix = 0
            hdl = []
            for i in range(len(error)):
                if not len(error[i]): continue
                hd = self.__get_error_seg__(error[i], histpl["label"][ix] + " (Uncertainty)", histpl["color"][ix])
                hdl.append(hd)
                ix+=1
            #self._ax.legend(handles = hdl, ncol = 2)

        elif self.ErrorBars:
            histpl["H"] = [self.__build__()]
            hts = dict(histpl)
            if "hatch" in histpl: del hts["hatch"]
            if "label" in histpl: del hts["label"]
            if "edgecolor" in histpl: del hts["edgecolor"]

            hts["alpha"] = 0.0
            hts["binticks"] = True
            hts["histtype"] = "errorbar"
            error = hep.histplot(**hts)
            hep.histplot(**histpl)
            self.__get_error_seg__(error[0])
        else: hep.histplot(**histpl)

        if not len(labels):
            self._ax.set_xlim(x_min, x_max, auto = True)
            self._ax.set_ylim(y_min, y_max, auto = True)
        self._ax.legend(loc = "upper right", ncol = 2 * (1 - 0.5*(not self.ErrorBars)))
        return {}

cdef class TH2F(BasePlotting):
    def __cinit__(self): self.ptr.prefix = b"TH2F"
    def __init__(self, inpt = None, **kwargs):
        self.Color = "plasma"
        if len(kwargs): inpt = {"data" : dict(kwargs)}
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["data"][i])
            except KeyError: continue
            except AttributeError: continue

    @property
    def yBins(self):
        if not self.ptr.variable_y_bins.size(): return self.ptr.y_bins
        else: return self.ptr.variable_y_bins

    @yBins.setter
    def yBins(self, val):
        if isinstance(val, int): self.ptr.y_bins = val
        elif isinstance(val, list): self.ptr.variable_y_bins = <vector[float]>(val)
        else: self.ptr.warning(b"Invalid Bins specified")

    @property
    def xBins(self):
        if not self.ptr.variable_x_bins.size(): return self.ptr.x_bins
        else: return self.ptr.variable_x_bins

    @xBins.setter
    def xBins(self, val):
        if isinstance(val, int): self.ptr.x_bins = val
        elif isinstance(val, list): self.ptr.variable_x_bins = <vector[float]>(val)
        else: self.ptr.warning(b"Invalid Bins specified")

    @property
    def xLabels(self): return as_basic_udict(&self.ptr.x_labels)
    @xLabels.setter
    def xLabels(self, dict val): as_umap(val, &self.ptr.x_labels)

    @property
    def yLabels(self): return as_basic_udict(&self.ptr.y_labels)
    @yLabels.setter
    def yLabels(self, dict val): as_umap(val, &self.ptr.y_labels)

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
        cdef dict xlabels = self.xLabels

        if len(xlabels): pass
        elif self.set_xmin: x_min = self.ptr.x_min
        elif not len(xlabels) and not len(self.xData): pass
        elif self.ptr.variable_x_bins.size(): x_min = self.ptr.variable_x_bins.front()
        else: x_min = self.ptr.get_min(b"x")

        if len(xlabels): pass
        elif self.set_xmax: x_max = self.ptr.x_max
        elif not len(xlabels) and not len(self.xData): pass
        elif self.ptr.variable_x_bins.size(): x_max = self.ptr.variable_x_bins.back()
        else: x_max = self.ptr.get_max(b"x")

        cdef float y_max, y_min
        cdef dict ylabels = self.yLabels

        if len(ylabels): pass
        elif self.set_ymin: y_min = self.ptr.y_min
        elif not len(ylabels) and not len(self.yData): pass
        elif self.ptr.variable_y_bins.size(): y_min = self.ptr.variable_y_bins.front()
        else: y_min = self.ptr.get_min(b"y")

        if len(ylabels): pass
        elif self.set_xmax: y_max = self.ptr.y_max
        elif not len(ylabels) and not len(self.yData): pass
        elif self.ptr.variable_x_bins.size(): y_max = self.ptr.variable_y_bins.back()
        else: y_max = self.ptr.get_max(b"y")


        if self.ptr.variable_x_bins.size(): ax_ = bh.axis.Variable(self.ptr.variable_x_bins)
        else: ax_ = bh.axis.Regular(self.ptr.x_bins, x_min, x_max)

        if self.ptr.variable_y_bins.size(): ay_ = bh.axis.Variable(self.ptr.variable_y_bins)
        else: ay_ = bh.axis.Regular(self.ptr.y_bins, y_min, y_max)

        h = bh.Histogram(ax_, ay_, storage = bh.storage.Weight())
        if not self.ptr.weights.size(): h.fill(self.ptr.x_data, self.ptr.y_data)
        else: h.fill(self.ptr.x_data, self.ptr.y_data, weight = self.ptr.weights)
        return h

    cdef dict __compile__(self, bool raw = False):
        cdef dict histpl = {}
        histpl["H"] = self.__build__()
        if len(self.Color): histpl["cmap"] = self.Color
        histpl["alpha"] = 1
        histpl["antialiased"] = True
        histpl["linewidth"] = 0
        histpl["zorder"] = -1
        histpl["cbarpad"] = 0.1
        hst = hep.hist2dplot(**histpl)
        cbar = hst.cbar
        cbar.set_label('Bin Content', loc = "center", rotation=270, labelpad = 20)

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
    def __cinit__(self): 
        self.ErrorShade = False
        self.ptr.prefix = b"TLine"

    def __init__(self, inpt = None, **kwargs):
        self.Lines = []
        self.Marker = ""
        if len(kwargs): inpt = {"data" : dict(kwargs)}
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

    @property
    def Alpha(self): return self.ptr.alpha

    @Alpha.setter
    def Alpha(self, float v): self.ptr.alpha = v

    cdef __error__(self, vector[float] xarr, vector[float] up, vector[float] low, str label = "Uncertainty", str color = "k"):
        cdef dict apl = {"step" : "post", "hatch" : "///", "alpha" : 0.15, "linewidth" : 0.0}
        if len(label): apl["label"] = label
        apl["facecolor"] = color if len(color) else None
        apl["edgecolor"] = ("k", 1.0)
        self.matpl.fill_between(xarr, low, up, **apl)

        apl = {"hatch" : "///", "step": "post", "alpha" : 0.00}
        return self.matpl.fill_between(xarr, low, up, **apl)

    cdef void factory(self):
        cdef dict coms = {}
        coms["linestyle"] = self.LineStyle
        if len(self.Color): coms["color"] = self.Color
        coms["marker"] = self.Marker
        coms["linewidth"] = self.LineWidth
        coms["label"] = self.Title
        coms["alpha"] = self.Alpha

        if not len(self.xData): return
        elif self.ErrorBars and self.ErrorShade:
            self.__error__(self.ptr.x_data, self.ptr.y_error_up, self.ptr.y_error_down, self.Title + " (unc.)", self.Color)
            self.matpl.plot(self.xData, self.yData, **coms)
        elif self.ErrorBars:
            self.ptr.build_error()
            coms["yerr"] = [self.yDataDown, self.yDataUp]
            coms["capsize"] = self.ptr.cap_size
            self.matpl.errorbar(self.xData, self.yData, **coms)
        elif len(self.yDataDown) == len(self.yDataUp) == len(self.xData):
            coms["yerr"] = [self.yDataDown, self.yDataUp]
            coms["capsize"] = self.ptr.cap_size
            self.matpl.errorbar(self.xData, self.yData, **coms)
        else: self.matpl.plot(self.xData, self.yData, **coms)

    cdef dict __compile__(self, bool raw = False):

        cdef float x_max, x_min
        if self.set_xmin: x_min = self.ptr.x_min
        elif not self.ptr.x_data.size(): x_min = 0
        else: x_min = self.ptr.get_min(b"x")

        if self.set_xmax: x_max = self.ptr.x_max
        elif not self.ptr.x_data.size(): x_max = 0
        else: x_max = self.ptr.get_max(b"x")

        cdef float y_max, y_min
        if self.set_ymin: y_min = self.ptr.y_min
        elif not self.ptr.y_data.size(): y_min = 0
        else: y_min = self.ptr.get_min(b"y")

        if self.set_ymax: y_max = self.ptr.y_max
        elif not self.ptr.y_data.size(): y_max = 0
        else: y_max = self.ptr.get_max(b"y")

        if not len(self.Lines) and not self.ptr.x_data.size(): return {}
        cdef TLine i

        cdef vector[float] dy = []
        cdef vector[float] dx = []
        for i in self.Lines: 
            i.factory()
            merge_data(&dy, &i.ptr.y_data)
            merge_data(&dx, &i.ptr.x_data)

        self._ax.tick_params(axis = "x", which = "minor", bottom = False)
        self.matpl.legend(loc = "best", ncol = 2 * (1 - 0.5*(not len(self.Lines) > 3)))
        self.factory()

        if not self.set_ymin: y_min = self.ptr.min(&dy);
        if not self.set_ymax: y_max = self.ptr.max(&dy); y_max = y_max*(1 + 0.1)
        if not self.set_xmin: x_min = self.ptr.min(&dx); 
        if not self.set_xmax: x_max = self.ptr.max(&dx); x_max = x_max*(1 + 0.1)

        if x_max > x_min: self.matpl.xlim(x_min, x_max)
        if y_max > y_min: self.matpl.ylim(y_min, y_max)
        self.matpl.title(self.Title)
        return {}

