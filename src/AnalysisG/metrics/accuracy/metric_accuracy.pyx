# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.roc cimport *

cdef tuple mx_index(vector[double]* sc):
    cdef int x, i
    cdef double s = 0
    for i in range(sc.size()):
        if s > sc.at(i): continue
        s = sc.at(i); x = i
    return (x, s)

cdef void get_data(AccuracyMetric vl, dict data, dict meta):
    cdef collector* cl = vl.cl
    cdef int epoch = meta[b"epoch"]
    cdef int kfold = meta[b"kfold"]
    cdef int ntops
    cdef tuple mx

    cdef vector[double] score
    cdef string model = meta[b"model_name"]
    cdef string mode = b""
    cdef string key
    cdef tools tl

    for key in data:
        if mode.size(): pass
        elif not mode.size() and tl.has_string(&key, b"evaluation"): mode = b"evaluation"
        elif not mode.size() and tl.has_string(&key, b"validation"): mode = b"validation"
        elif not mode.size() and tl.has_string(&key, b"training"):   mode = b"training"
        else: continue

        ntops = data[b"event_accuracy_" + mode + b".ntop_truth.ntop_truth"]
        score = data[b"event_accuracy_" + mode + b".ntop_scores.ntop_scores"]
        mx    = mx_index(&score)

        if   tl.has_string(&key, b"ntop_truth" ): cl.add_ntop_truth(mode, model, epoch, kfold, ntops)
        elif tl.has_string(&key, b"edge"       ): cl.add_ntop_edge_accuracy(mode, model, epoch, kfold, ntops, <double>(data[key]))
        elif tl.has_string(&key, b"ntop_scores"):
            cl.add_ntop_scores(mode, model, epoch, kfold, &score)
            cl.add_ntru_ntop_scores(mode, model, epoch, kfold, ntops, int(mx[0]), <double>(mx[1]))

cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):
        self.root_leaves = {
            "event_accuracy_training"   : ["ntop_truth", "ntop_scores", "edge"],
            "event_accuracy_validation" : ["ntop_truth", "ntop_scores", "edge"],
            "event_accuracy_evaluation" : ["ntop_truth", "ntop_scores", "edge"],
        }

        self.root_fx = {
            "event_accuracy_training"   : get_data,
            "event_accuracy_validation" : get_data,
            "event_accuracy_evaluation" : get_data,
    
        }

        self.mtx = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.mtx)
        self.cl  = new collector()
        self.default_plt = None
        self.auc = {}

    def Postprocessing(self):
        cdef ROC rc
        cdef cdata_t* px
        self.cl.get_plts()

        cdef vector[string] model_names = self.cl.model_names
        cdef vector[string] modes_names = self.cl.modes
        cdef vector[int]    epochs      = self.cl.epochs
        cdef vector[int]    kfolds      = self.cl.kfolds

        cdef TLine tl, tm
        cdef str name_, mode_
        cdef string name, mode
        cdef int ep, kf

        cdef dict colx = {}
        cdef dict lines = {}

        tm = TLine()
        for ep in epochs:
            for name in model_names:
                name_ = env(name)
                if name_ not in colx:
                    colx[name_] = tm.Color
                    tm.Color = ""
                rc = ROC()
                rc.xBins = 100
                rc.default_plt = self.default_plt
                rc.OutputDirectory = "./figures/epoch-" + str(ep) + "/" + env(name)
                rc.Title = "Top Multiplicity Classification: (" + env(name) + " @ Epoch-" + str(ep) +")"
                rc.Filename = "ntops"
                for kf in kfolds:
                    for mode in modes_names:
                        px = self.cl.get_mode(name, mode, ep, kf)
                        if px == NULL: continue
                        rc.rx.build_ROC(mode, kf, &px.ntops_truth, &px.ntop_score)
                rc.__compile__()
                if name_ not in self.auc: self.auc[name_] = {}
                self.auc[name_][ep] = rc.auc

                for mode_ in rc.auc:
                    for cls_ in rc.auc[mode_]:
                        if "::" not in str(cls_): continue
                        clx = "cls::" + cls_.split("::")[1]
                        if clx not in lines: lines[clx] = {}
                        if name_ not in lines[clx]: lines[clx][name_] = {}
                        if mode_ not in lines[clx][name_]: lines[clx][name_][mode_] = {}
                        if ep in lines[clx][name_][mode_]: continue
                        lines[clx][name_][mode_][ep] = [rc.auc[mode_][clx + "::avg"], rc.auc[mode_][clx + "::stdev"]]
  
        cdef dict cols = {"training" : "-", "validation" : "--", "evaluation" : ":"}
        for cls_ in lines:
            tm = TLine()
            linex = []
            for name_ in lines[cls_]:
                for mode_ in lines[cls_][name_]:
                    epx = sorted(lines[cls_][name_][mode_])
                    dax = lines[cls_][name_][mode_]
                    tl = TLine()
                    tl.LineStyle = cols[mode_] 
                    tl.Color = colx[name_]; tl.Alpha = 1.0
                    tl.Title = name_ + " (" + mode_ + ")"

                    tl.xData     = epx
                    tl.yData     = [dax[ep][0] for ep in epx]
                    tl.yDataDown = [dax[ep][0] - dax[ep][1] for ep in epx]
                    tl.yDataUp   = [dax[ep][0] + dax[ep][1] for ep in epx]
                    tl.ErrorShade = True; tl.ErrorBars = True
                    if self.default_plt is None: pass
                    else: self.default_plt(tl)
                    linex.append(tl)

            if self.default_plt is None: pass
            else: self.default_plt(tm)
            tm.Title = "Model Performance Top Multiplicity (n-Top: " + cls_.split("::")[1] + ")"
            tm.xTitle = "Epochs"
            tm.yTitle = "AUC"
            tm.Lines = linex
            tm.xMin = 0; tm.xMax = max(epochs)+1
            tm.yMin = 0; tm.yMax = 1
            tm.OutputDirectory = "./figures/summary"
            tm.Filename = "ntop-" + cls_.split("::")[1]
            tm.SaveFigure() 

        f = open("./figures/summary/roc.txt", "w")
        f.write(str(self.roc))
        f.close()

