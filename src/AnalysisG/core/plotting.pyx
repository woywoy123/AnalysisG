# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.string cimport string

from AnalysisG.core.tools cimport env
from AnalysisG.core.plotting cimport plotting

import boost_histogram as bh
import matplotlib as plt
import mplhep as hep

cdef class BasePlotting:
    def __cinit__(self):
        self.ptr = new plotting()
        self.matpl = plt
        self.__resetplt__()

    cdef void __figure__(self):
        cdef dict com = {}
        com["figsize"] = (self.ptr.xscaling, self.ptr.yscaling)
        self._fig, self._ax = self.matpl.subplots(**com)
        self._ax.set_autoscale_on(self.ptr.auto_scale)

        com = {}
        com["font.size"] = self.font_size
        com["axes.labelsize"] = self.axis_size
        com["legend.fontsize"] = self.font_size
        com["figure.titlesize"] = self.title_size
        com["text.usetex"] = self.use_latex
        self.matpl.rcParams.update(com)

    cdef void __resetplt__(self):
        self.matpl.clf()
        self.matpl.cla()
        self.matpl.close("all")

        self.matpl = plt
        self.matpl.rcdefaults()
        self.__makefigure__()

    cdef void __compile__(self): pass

    def __dealloc__(self): del self.ptr
    def __init__(self): pass

    @property
    def Filename(self): return self.ptr.filename

    @Filename.setter
    def Filename(self, str val): self.ptr.filename = env(val)

    @property
    def OutputDirectory(self): return self.ptr.output_path

    @OutputDirectory.setter
    def OutputDirectory(self, str val): self.ptr.output_path = env(val)

    @property
    def xMin(self): return self.ptr.x_min

    @xMin.setter
    def xMin(self, float val): self.ptr.x_min = val

    @property
    def yMin(self): return self.ptr.y_min

    @yMin.setter
    def yMin(self, float val): self.ptr.y_min = val

    @property
    def xMax(self): return self.ptr.x_max

    @xMax.setter
    def xMax(self, float val): self.ptr.x_max = val

    @property
    def yMax(self): return self.ptr.y_max

    @yMax.setter
    def yMax(self, float val): self.ptr.y_max = val

    def SaveFigure(self):
        cdef string out = self.ptr.build_path()
        self.__compile__()


cdef class TH1F(BasePlotting):

    def __init__(self): pass

    @property
    def xData(self): return self.ptr.x_data;

    @xData.setter
    def xData(self, list val): self.ptr.x_data = <vector[float]>(val)

    @property
    def xBins(self): return self.ptr.x_bins

    @xBins.setter
    def xBins(self, int val): self.ptr.x_bins = val

    cdef void __compile__(self):
        print("...")


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

