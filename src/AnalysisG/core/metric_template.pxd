# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.core.io cimport *
from cython.operator cimport dereference as deref

cdef extern from "<templates/metric_template.h>" nogil:
    cdef cppclass metric_template(tools, notification):
        metric_template() except+ nogil
        string name
        map[string, string] run_names
        vector[string] variables

cdef inline bool finder(string* fname, vector[string]* kfolds, vector[string]* epochs) nogil:
    cdef tools tl
    cdef string ix
    cdef bool found_k = False
    cdef bool found_e = False
    for ix in deref(kfolds):
        if not tl.ends_with(fname, string(b"kfold-") + ix + string(b".root")): continue
        found_k = True; break

    for ix in deref(epochs):
        if not tl.has_string(fname, string(b"epoch-") + ix + string(b"/")): continue
        found_e = True; break
    return (kfolds.size() == 0 or found_k)*(epochs.size() == 0 or found_e)

cdef class MetricTemplate(Tools):
    cdef metric_template* mtx
    cdef public dict root_leaves
    cdef public dict root_fx
