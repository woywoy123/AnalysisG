# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.roc cimport *

cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):
        cdef list ev_ptr  = ["particle_"   + i for i in ["pt", "eta", "phi", "energy", "chn"]]
        cdef list tps_tru = ["tops_truth_" + i for i in ["pt", "eta", "phi", "mass"  , "chn"]]
        cdef list tps_npr = ["tops_pr_"    + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_upr = ["tops_upr_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_nom = ["tops_nom_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list evnt_dt = ["process_idx", "average_edge", "ntops_scores", "dsid", "ntops_tru", "ntops_prd"]

        self.root_leaves = {
            "accuracy_training" : ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt,
            #            "accuracy_validation;1" : ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru,
        }

        self.root_fx = {
            "accuracy_training" : get_data,
            #            "accuracy_validation;1" : get_data,
        }

        self.mtx = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.mtx)
        self.mcl = new collector() 
        self.default_plt = None

    def __delloc__(self):
        self.mtx = NULL; 
        del self.mtr
        del self.mcl

    def Postprocessing(self):
        pass

#        cdef ROC rc
#        cdef cdata_t* px
#        self.cl.get_plts()
#
#        cdef vector[string] model_names = self.cl.model_names
#        cdef vector[string] modes_names = self.cl.modes
#        cdef vector[int]    epochs      = self.cl.epochs
#        cdef vector[int]    kfolds      = self.cl.kfolds
#
#        cdef TLine tl, tm
#        cdef str name_, mode_
#        cdef string name, mode
#        cdef int ep, kf
#
#        cdef dict colx = {}
#        cdef dict lines = {}
#
#        tm = TLine()
#        for ep in epochs:
#            for name in model_names:
#                name_ = env(name)
#                if name_ not in colx:
#                    colx[name_] = tm.Color
#                    tm.Color = ""
#                rc = ROC()
#                rc.xBins = 100
#                rc.default_plt = self.default_plt
#                rc.OutputDirectory = "./figures/epoch-" + str(ep) + "/" + env(name)
#                rc.Title = "Top Multiplicity Classification: (" + env(name) + " @ Epoch-" + str(ep) +")"
#                rc.Filename = "ntops"
#                for kf in kfolds:
#                    for mode in modes_names:
#                        px = self.cl.get_mode(name, mode, ep, kf)
#                        if px == NULL: continue
#                        rc.rx.build_ROC(mode, kf, &px.ntops_truth, &px.ntop_score)
#                rc.__compile__()
#                if name_ not in self.auc: self.auc[name_] = {}
#                self.auc[name_][ep] = rc.auc
#
#                for mode_ in rc.auc:
#                    for cls_ in rc.auc[mode_]:
#                        if "::" not in str(cls_): continue
#                        clx = "cls::" + cls_.split("::")[1]
#                        if clx not in lines: lines[clx] = {}
#                        if name_ not in lines[clx]: lines[clx][name_] = {}
#                        if mode_ not in lines[clx][name_]: lines[clx][name_][mode_] = {}
#                        if ep in lines[clx][name_][mode_]: continue
#                        lines[clx][name_][mode_][ep] = [rc.auc[mode_][clx + "::avg"], rc.auc[mode_][clx + "::stdev"]]
#  
#        cdef dict cols = {"training" : "-", "validation" : "--", "evaluation" : ":"}
#        for cls_ in lines:
#            tm = TLine()
#            linex = []
#            for name_ in lines[cls_]:
#                for mode_ in lines[cls_][name_]:
#                    epx = sorted(lines[cls_][name_][mode_])
#                    dax = lines[cls_][name_][mode_]
#                    tl = TLine()
#                    tl.LineStyle = cols[mode_] 
#                    tl.Color = colx[name_]; tl.Alpha = 1.0
#                    tl.Title = name_ + " (" + mode_ + ")"
#
#                    tl.xData     = epx
#                    tl.yData     = [dax[ep][0] for ep in epx]
#                    tl.yDataDown = [dax[ep][0] - dax[ep][1] for ep in epx]
#                    tl.yDataUp   = [dax[ep][0] + dax[ep][1] for ep in epx]
#                    tl.ErrorShade = True; tl.ErrorBars = True
#                    if self.default_plt is None: pass
#                    else: self.default_plt(tl)
#                    linex.append(tl)
#
#            if self.default_plt is None: pass
#            else: self.default_plt(tm)
#            tm.Title = "Model Performance Top Multiplicity (n-Top: " + cls_.split("::")[1] + ")"
#            tm.xTitle = "Epochs"
#            tm.yTitle = "AUC"
#            tm.Lines = linex
#            tm.xMin = 0; tm.xMax = max(epochs)+1
#            tm.yMin = 0; tm.yMax = 1
#            tm.OutputDirectory = "./figures/summary"
#            tm.Filename = "ntop-" + cls_.split("::")[1]
#            tm.SaveFigure() 
#
#        f = open("./figures/summary/roc.txt", "w")
#        f.write(str(self.roc))
#        f.close()
#
