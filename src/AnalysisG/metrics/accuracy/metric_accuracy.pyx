# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.tools cimport *




cdef void get_truth(AccuracyMetric vl, dict data):
    cdef int epoch = data[b"epoch"]
    cdef int kfold = data[b"kfold"]
    if epoch > 1: return

    cdef string var
    cdef string mode
    if   b"event_accuracy_evaluation.ntop_truth.ntop_truth" in data: var = b"ntop_truth"; mode = b"evaluation"
    elif b"event_accuracy_validation.ntop_truth.ntop_truth" in data: var = b"ntop_truth"; mode = b"validation"
    elif b"event_accuracy_training.ntop_truth.ntop_truth"   in data: var = b"ntop_truth"; mode = b"training"

    cdef int val = data[b"event_accuracy_" + mode + b"." + var + "." + var]
    vl.truth_data[var][mode].kfold.push_back(kfold)
    vl.truth_data[var][mode].ntops.push_back(val)


cdef void get_ntop(AccuracyMetric vl, dict data):
    cdef string model = data[b"model_name"]
    cdef int epoch = data[b"epoch"]
    cdef int kfold = data[b"kfold"]

    cdef string var
    cdef string mode
    if   b"event_accuracy_evaluation.ntop_scores.ntop_scores" in data: var = b"ntop_scores"; mode = b"evaluation"
    elif b"event_accuracy_validation.ntop_scores.ntop_scores" in data: var = b"ntop_scores"; mode = b"validation"
    elif b"event_accuracy_training.ntop_scores.ntop_scores"   in data: var = b"ntop_scores"; mode = b"training"
    cdef vector[double] val = data[b"event_accuracy_" + mode + b"." + var + "." + var]
    

    








cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):

        self.root_leaves = {
            "event_accuracy_training"   : ["ntop_truth", "ntop_scores.ntop_scores"],
            "event_accuracy_validation" : ["ntop_truth", "ntop_scores.ntop_scores"],
            "event_accuracy_evaluation" : ["ntop_truth", "ntop_scores.ntop_scores"],
        }

        self.root_fx = {
            "event_accuracy_training.ntop_truth"   : get_truth,
            "event_accuracy_validation.ntop_truth" : get_truth,
            "event_accuracy_evaluation.ntop_truth" : get_truth,
        







        }

        self.mtx = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.mtx)

    def Postprocessing(self): pass
