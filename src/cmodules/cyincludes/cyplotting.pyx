# distutils: language = c++
# cython: language_level = 3

from cython.operator cimport dereference
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

from cyplotting cimport CyPlotting, CyMetric
from cyplotstruct cimport *
from cymetrics cimport *

from cytypes cimport metric_t
from cytools cimport env, enc

cimport numpy
cdef numpy.ndarray arr
import numpy as np

from torchmetrics import ROC, AUROC, ConfusionMatrix
import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import torch
import os

cdef class BasePlotting:
    cdef CyPlotting* ptr
    cdef paint_t* pn
    cdef figure_t* fig
    cdef io_t* io

    cdef matpl
    cdef _ax
    cdef _fig
    cdef dict data_axis

    cdef axis_t x
    cdef axis_t y

    def __cinit__(self):
        self.ptr = new CyPlotting()
        self.pn = &self.ptr.painter_params
        self.io = &self.ptr.file_params
        self.fig = &self.ptr.figure_params
        self.data_axis = {}
        self.matpl = plt

        self.x = axis_t()
        self.y = axis_t()

        self.x.dim = b"x-axis"
        self.y.dim = b"y-axis"

        self.x.title = b"x-axis"
        self.y.title = b"y-axis"

    def __dealloc__(self): del self.ptr
    def __init__(self): pass
    cdef __atlas__(self):
        cdef paint_t* p = self.pn
        hep.atlas.text(loc = p.atlas_loc)
        cdef dict dic = {}
        if p.atlas_data: dic["data"] = p.atlas_data
        if p.atlas_com > 0: dic["com"] = p.atlas_com
        if p.atlas_lumi > 0: dic["lumi"] = p.atlas_lumi
        if p.atlas_year > 0: dic["year"] = int(p.atlas_year)
        hep.atlas.label(**dic)
        self.matplt.style.use(hep.style.ATLAS)

    cdef __root__(self):
        self.matplt.style.use(hep.style.ROOT)

    cdef __style__(self):
        if self.pn.atlas_style: self.__atlas__()
        if self.pn.root_style: self.__root__()

    cdef __makefigure__(self):
        cdef dict com = {}
        com["figsize"] = (self.xScaling, self.yScaling)
        self._fig, self._ax = self.matpl.subplots(**com)
        self._ax.set_autoscale_on(self.autoscale)

        com = {}
        com["font.size"] = self.FontSize
        com["axes.labelsize"] = self.LabelSize
        com["legend.fontsize"] = self.LegendSize
        com["figure.titlesize"] = self.TitleSize
        com["text.usetex"] = self.LaTeX
        self.matpl.rcParams.update(com)

    cdef __resetplt__(self):
        self.matpl.close("all")
        self.matpl = plt
        self.matpl.rcdefaults()
        self.__makefigure__()


    cdef add_data(self, axis_t axis, data):
        self.data_axis[env(axis.dim)] = data

    def __precompile__(self): pass
    def __compile__(self): pass
    def __postcompile__(self): pass

    def SaveFigure(self, dirc = None):
        if dirc is not None: pass
        else: dirc = env(self.io.outputdir)

        try: dirc = os.path.abspath(dirc)
        except: pass

        try: os.makedirs(dirc)
        except FileExistsError: pass

        self.__precompile__()
        self.__style__()
        self.__makefigure__()

        self.__compile__()
        self._ax.set_title(self.Title)
        self.__postcompile__()

        try: self.matpl.xlabel(env(self.x.title), size = self.LabelSize)
        except AttributeError: pass
        try: self.matpl.ylabel(env(self.y.title), size = self.LabelSize)
        except AttributeError: pass

        if dirc.endswith("/"): pass
        else: dirc += "/"

        self.matpl.tight_layout()
        self.matpl.savefig(dirc + self.Filename)

    @property
    def Filename(self): return env(self.io.filename)
    @Filename.setter
    def Filename(self, val):
        if val.endswith(".png"): pass
        else: val += ".png"
        self.io.filename = enc(val)

    @property
    def FontSize(self): return self.pn.font_size
    @FontSize.setter
    def FontSize(self, val): self.pn.font_size = val

    @property
    def LabelSize(self): return self.pn.label_size
    @LabelSize.setter
    def LabelSize(self, val): self.pn.label_size = val

    @property
    def LaTeX(self): return self.pn.latex
    @LaTeX.setter
    def LaTeX(self, val): self.pn.latex = val

    @property
    def TitleSize(self): return self.pn.title_size
    @TitleSize.setter
    def TitleSize(self, val): self.pn.title_size = val

    @property
    def LegendSize(self): return self.pn.legend_size
    @LegendSize.setter
    def LegendSize(self, val): self.pn.legend_size = val

    @property
    def LegendLoc(self): return env(self.pn.legend_loc)
    @LegendLoc.setter
    def LegendLoc(self, val): self.pn.legend_loc = enc(val)


    @property
    def NEvents(self): return self.pn.n_events
    @NEvents.setter
    def NEvents(self, int val): self.pn.n_events = val

    @property
    def xScaling(self): return self.pn.xscaling
    @xScaling.setter
    def xScaling(self, float val): self.pn.xscaling = val

    @property
    def yScaling(self): return self.pn.yscaling
    @yScaling.setter
    def yScaling(self, float val): self.pn.yscaling = val

    @property
    def Color(self): return str(self.pn.color)
    @Color.setter
    def Color(self, str val): self.pn.color = env(val)

    @property
    def autoscale(self): return self.pn.autoscale
    @autoscale.setter
    def autoscale(self, bool val): self.pn.autoscale = val

    @property
    def Title(self): return env(self.fig.title)
    @Title.setter
    def Title(self, val): self.fig.title = enc(val)

    @property
    def xTitle(self): return env(self.x.title)
    @xTitle.setter
    def xTitle(self, val): self.x.title = enc(val)

    @property
    def yTitle(self): return env(self.y.title)
    @yTitle.setter
    def yTitle(self, val): self.y.title = enc(val)


    @property
    def Style(self):
        if self.pn.atlas_style: return "ATLAS"
        if self.pn.root_style: return "ROOT"
        if self.pn.mpl_style: return "MPL"
        return None

    @Style.setter
    def Style(self, val):
        cdef str x = val.upper()
        if x == "ATLAS": self.pn.atlas_style = True
        if x == "ROOT": self.pn.root_style = True
        if x == "MPL": self.pn.mpl_style = True


    @property
    def xMin(self):
        if self.x.start == self.x.end: return None
        else: return self.x.start

    @xMin.setter
    def xMin(self, val): self.x.start = val

    @property
    def xMax(self):
        if self.x.start == self.x.end: return None
        else: return self.x.end

    @xMax.setter
    def xMax(self, val): self.x.end = val


cdef class TH1F(BasePlotting):
    def __init__(self):
        self.fig.histogram = True


    cdef __fixdata__(self):
        arr = np.array(self.xData)
        min_, max_ = self.xMin, self.xMax
        if max_ is None: self.xMax = arr.max()
        if min_ is None: self.xMin = arr.min()
        self.xBins = len(arr)
        self.xData = arr

    def __precompile__(self):
        if self.xBins is not None: pass
        else: self.__fixdata__()

    def __compile__(self):
        cdef dict com = {}
        com["bins"] = self.xBins
        com["start"] = self.xMin
        com["stop"] = self.xMax
        com["underflow"] = self.UnderFlow
        com["overflow"] = self.OverFlow

        hist = bh.Histogram(bh.axis.Regular(**com))
        hist.fill(self.xData)
        hep.histplot(hist)

    @property
    def xData(self):
        try: return self.data_axis[env(self.x.dim)]
        except KeyError: return None

    @xData.setter
    def xData(self, inpt):
        self.add_data(self.x, inpt)

    @property
    def xBins(self):
        if not self.x.bins: return None
        return self.x.bins
    @xBins.setter
    def xBins(self, val): self.x.bins = val

    @property
    def UnderFlow(self): return self.x.underflow
    @UnderFlow.setter
    def UnderFlow(self, val): self.x.underflow = val

    @property
    def OverFlow(self): return self.x.overflow
    @OverFlow.setter
    def OverFlow(self, val): self.x.overflow = val




cdef class MetricPlots(BasePlotting):

    cdef public int epoch
    cdef public str path
    cdef bool triggered
    cdef CyMetric* mtrk

    cdef ROC(self, roc_t* inpt):
        pre = torch.tensor(inpt.pred)
        tru = torch.tensor(inpt.truth, dtype = torch.int).view(-1)

        cdef dict coms = {"task" : "multiclass", "num_classes" : pre.size()[1]}
        roc = ROC(**coms)
        auc = AUROC(**coms)
        fpr, tpr, thres = roc(pre, tru)
        for cls in range(len(fpr)):
            dereference(inpt).tpr[cls] = tpr[cls].tolist()
            dereference(inpt).fpr[cls] = fpr[cls].tolist()
            dereference(inpt).threshold[cls] = thres[cls].tolist()

        auc_t = auc(pre, tru).view(-1)
        for cls in range(len(auc_t)): dereference(inpt).auc[cls+1] = auc_t[cls].item()

        conf = ConfusionMatrix(**coms)
        dereference(inpt).confusion = conf(pre, tru).tolist()
        dereference(inpt).truth.clear()
        dereference(inpt).pred.clear()

    cpdef void AddMetrics(self, map[string, metric_t] val, string mode):
        if not self.triggered: self.mtrk = new CyMetric()
        self.triggered = True
        self.mtrk.current_epoch = self.epoch
        self.mtrk.AddMetric(&val, mode)

        cdef roc_t* itr
        for itr in self.mtrk.FetchROC(): self.ROC(itr)

    cpdef void ReleasePlots(self, str paths):
        self.mtrk.outpath = enc(paths)
        cdef map[string, abstract_plot] output
        self.mtrk.BuildPlots(&output)


