# distutils: language=c++
# cython: language_level=3
from cython.parallel import prange

cdef extern from "<tools/merge_cast.h>":
    cdef void merge_data(vector[plt_roc_t*]* oux, vector[plt_roc_t*]* inx) except+ nogil

cdef tuple mx_index(vector[double]* sc):
    cdef int x, i
    cdef double s = 0
    for i in range(sc.size()):
        if s > sc.at(i): continue
        s = sc.at(i); x = i
    return (x, s)

cdef void get_data(AccuracyMetric vl, dict data, dict meta):
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

        if   tl.has_string(&key, b"ntop_truth" ): vl.event_level[mode][epoch].ntops_truth[model][kfold].push_back(ntops)
        elif tl.has_string(&key, b"edge"       ): vl.event_level[mode][epoch].edge_scores[model][ntops].push_back(data[key])
        elif tl.has_string(&key, b"ntop_scores"): 
            vl.event_level[mode][epoch].ntop_score[model][kfold].push_back(score)
            vl.event_level[mode][epoch].ntru_npred_ntop[model][ntops][mx[0]].push_back(mx[1])
        else: continue

cdef vector[plt_roc_t*] make_roc(plt_roc_t* out, modelx_t* inx):
    cdef plt_roc_t* mt
    cdef vector[plt_roc_t*] mxo
    cdef pair[string, map[int, vector[int]]] itx
    cdef pair[int, vector[vector[double]]]  its

    for itx in inx.ntops_truth:
        for its in inx.ntop_score[itx.first]:
            mt = new plt_roc_t()
            mt.kfold = its.first
            mt.model = itx.first + b"@" + out.model
            merge_data(&mt.scores, &its.second)
            mt.variable = out.variable 
            merge_data(&mt.truth, &itx.second[its.first])
            mxo.push_back(mt)
    return mxo


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

    def Postprocessing(self):
        cdef int e, k
        cdef plt_roc_t rx
        cdef plt_roc_t* rxp
        cdef vector[int] epochs

        cdef vector[plt_roc_t*] data_o
        cdef pair[string, map[int, modelx_t]] itm

        cdef pair[int, vector[plt_roc_t*]] eitr
        cdef map[int, vector[plt_roc_t*]] data_i

        cdef ROC rxc
        cdef list plts = []
        cdef map[int, plotting*] plts_ptr


        for itm in self.event_level:
            epochs = <vector[int]>(sorted(list(set(list(itm.second)))))
            for e in epochs:
                rx = plt_roc_t()
                rx.epoch = e
                rx.model = itm.first
                rx.variable = b"Top Multiplicity Performance"
                data_o = make_roc(&rx, &itm.second[e])
                merge_data(&data_i[e], &data_o)
                print(data_i[e].size())

        epochs = [eitr.first for eitr in data_i]
        for e in epochs:
            rxc = ROC()
            plts_ptr[e] = rxc.ptr
            plts.append(rxc)
        print(epochs)

        for e in prange(epochs.size(), nogil = True, num_threads = epochs.size()): 
            e = epochs[e]
            for k in range(data_i[e].size()):
                rxp = data_i[e][k]
                plts_ptr[e].build_ROC(rxp.model, rxp.kfold, &rxp.truth, &rxp.scores)

        for rxc in plts: rxc.__compile__()
