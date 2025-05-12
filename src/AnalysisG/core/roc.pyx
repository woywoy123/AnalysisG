# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool, float
from libcpp.map cimport map

from AnalysisG.core.tools cimport *
from cython.operator cimport dereference as dref
import numpy
import random
import pathlib
import pickle

cdef class ROC(TLine):
    def __cinit__(self):
        self.rx = new roc()
        self.num_cls = 0
        self.inits = True
        self.verbose = False
        self.ptr.prefix = b"ROC Curve"
        try: from sklearn import metrics; return
        except: self.inits = False
        self.ptr.warning(b"Failed to import sklearn.")

    def __dealloc__(self): del self.rx
    def __init__(self, inpt = None, **kwargs):
        self.default_plt = None
        self.Marker = ""
        self.auc    = {}
        self.Lines  = []
        self.xTitle = "False Positive Rate"
        self.yTitle = "True Positive Rate"
        self.xMax = 1
        self.yMax = 1
        self.xMin = 0
        self.yMin = 0

    cdef void factory(self): return

    cdef dict __compile__(self, bool raw = False):
        if not self.inits: return {}
        if not self.rx.roc_data.size(): return {}
        from sklearn.metrics import roc_curve, auc

        cdef str title
        cdef vector[roc_t*] points = self.rx.get_ROC()
        cdef vector[double] tpr_
        cdef vector[double] fpr_
        cdef double auc_

        cdef dict tprs = {}
        cdef dict aucs = {}
        cdef dict fprs = {}

        cdef int i, j, kf
        for i in range(points.size()):
            self.num_cls = points[i].cls
            title = env(points[i].model)
            kf    = points[i].kfold

            data  = numpy.array(dref(points[i].scores))
            truth = numpy.array(dref(points[i].truth))
            if title not in tprs:
                fprs[title] = {cl : numpy.linspace(0, 1, self.xBins) for cl in range(self.num_cls)}
                self.auc[title] = {}
                tprs[title] = {}
                aucs[title] = {}

            if kf not in tprs[title]:
                tprs[title][kf]     = {cl : None for cl in range(self.num_cls)}
                aucs[title][kf]     = {cl : None for cl in range(self.num_cls)}
                self.auc[title][kf] = {cl : None for cl in range(self.num_cls)}

            for j in range(self.num_cls):
                fpr, tpr, _ = roc_curve(truth[:, j], data[:, j])
                tpr_ = <vector[double]>(tpr)
                fpr_ = <vector[double]>(fpr)
                auc_ = auc(fpr, tpr)

                points[i]._auc.push_back(auc_)
                points[i].tpr_.push_back(tpr_)
                points[i].fpr_.push_back(fpr_)
                interp_tpr = numpy.interp(fprs[title][j], fpr, tpr)
                interp_tpr[0] = 0.0

                tprs[title][kf][j] = interp_tpr
                aucs[title][kf][j] = auc_
                self.auc[title][kf][j] = auc_
       
        cdef dict lines = {}
        cdef dict cols = {"training" : "red", "validation" : "blue", "evaluation" : "green"}
         
        cdef TLine tl
        for title in tprs:
            for i in range(self.num_cls):
                avg_tpr = numpy.mean([tprs[title][kf][i] for kf in tprs[title]], axis = 0)
                std_tpr = numpy.std( [tprs[title][kf][i] for kf in tprs[title]], axis = 0)
                upr_tpr = numpy.minimum(avg_tpr + std_tpr, 1)
                lrw_tpr = numpy.maximum(avg_tpr - std_tpr, 0)

                avg_tpr[-1] = 1.0
                avg_auc = auc(fprs[title][i], avg_tpr)
                std_auc = numpy.std([aucs[title][kf][i] for kf in aucs[title]])

                self.auc[title]["cls::" + str(i) + "::avg"  ] = avg_auc
                self.auc[title]["cls::" + str(i) + "::stdev"] = std_auc

                tl = TLine()
                tl.xMax = 1; tl.xMin = 0
                tl.yMax = 1; tl.yMin = 0
                tl.Title      = title + " AUC: " + str(round(avg_auc, 5)) + "$\\pm$" + str(round(std_auc, 5))
                tl.Color      = cols[title]
                tl.xData      = fprs[title][i].tolist()
                tl.yData      = avg_tpr.tolist()
                tl.yDataDown  = lrw_tpr.tolist()
                tl.yDataUp    = upr_tpr.tolist()
                tl.ErrorBars  = True
                tl.ErrorShade = True
                if self.default_plt is None: pass
                else: self.default_plt(tl)

                if i not in lines: lines[i] = {}
                lines[i][title] = tl

        for i in lines:
            tl = TLine()
            tl.Title = self.Title
            tl.xTitle = self.xTitle; tl.yTitle = self.yTitle
            tl.Lines = list(lines[i].values())
            if self.default_plt is None: pass
            else: self.default_plt(tl)
            tl.xMax = self.xMax; tl.xMin = self.xMin
            tl.yMax = self.yMax; tl.yMin = self.yMin
            tl.OutputDirectory = self.OutputDirectory
            tl.Filename = self.Filename + "_cls-" + str(i)
            tl.SaveFigure()
        return {}




    @property
    def xBins(self): return self.ptr.x_bins
    @xBins.setter
    def xBins(self, int val): self.ptr.x_bins = val

    @property
    def Scores(self): return None
    @property
    def Truth(self): return None

    @Scores.setter
    def Scores(self, val):
        cdef vector[vector[double]] data
        cdef vector[int]* vc = NULL

        if not self.inits: return
        if isinstance(val, list):
            data = <vector[vector[double]]>(val) 
            self.rx.build_ROC(b"name", -1, vc, &data)
        elif isinstance(val, dict):
            for k in val:
                if isinstance(k, str) and isinstance(val[k], list): 
                    data = <vector[vector[double]]>(val[k]) 
                    self.rx.build_ROC(enc(k), -1, vc, &data)
                elif isinstance(k, int) and isinstance(val[k], list):
                    data = <vector[vector[double]]>(val[k]) 
                    self.rx.build_ROC(enc(k), -1, vc, &data)
                elif isinstance(k, str) and isinstance(val[k], tuple):
                    data = <vector[vector[double]]>(val[k][1]) 
                    self.rx.build_ROC(enc(k), int(val[k][0]), vc, &data)
                else: return
        elif isinstance(val, tuple):
            data = <vector[vector[double]]>(val[2])
            self.rx.build_ROC(enc(val[0]), int(val[1]), <vector[int]*>(NULL), &data)
        else: self.failure(b"Expected: dict(str, tuple(int, list[list[float]]")

    @Truth.setter
    def Truth(self, val): 
        cdef vector[int] data
        cdef vector[vector[double]]* vc = NULL

        if not self.inits: return
        if isinstance(val, list):
            data = <vector[int]>(val)
            self.rx.build_ROC(b"name", -1, &data, vc)
        elif isinstance(val, dict):
            for k in val:
                if isinstance(k, str) and isinstance(val[k], list): 
                    data = <vector[int]>(val[k])
                    self.rx.build_ROC(enc(k), -1, &data, vc)
                elif isinstance(k, int) and isinstance(val[k], list):
                    data = <vector[int]>(val[k])
                    self.rx.build_ROC(enc(k), -1, &data, vc)
                elif isinstance(k, str) and isinstance(val[k], tuple):
                    data = <vector[int]>(val[k][1])
                    self.rx.build_ROC(enc(k), int(val[k][0]), &data, vc)
        elif isinstance(val, tuple):
            data = <vector[int]>(val[2])
            self.rx.build_ROC(enc(val[0]), int(val[1]), &data, vc)
        else: self.failure(b"Expected: dict(str, tuple(int, list[list[float]]")

    @property
    def Titles(self): return self.Lines
    @Titles.setter
    def Titles(self, list val): self.Lines = val

    @property
    def AUC(self): return self.auc

    @property
    def xData(self): return None
    @xData.setter
    def xData(self, val): self.ptr.warning(b"Wrong Input. Use Scores")

    @property
    def yData(self): return None
    @yData.setter
    def yData(self, val): self.ptr.warning(b"Wrong Input. Use Truth")


