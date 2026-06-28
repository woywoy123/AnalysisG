# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.core.io cimport *
from cython.parallel import prange
from cython.operator cimport dereference as deref

cdef extern from "<templates/metric_template.h>" nogil:
    cdef cppclass metric_template(tools, notification):
        metric_template() except+ nogil
        string name
        map[string, string] run_names
        vector[string] variables

cdef inline bool finder(string* fname, vector[string]* kfolds, vector[string]* epochs, string* prefx) nogil:
    cdef tools tl
    cdef string ix
    cdef string tmp
    cdef string pfx = deref(prefx)
    if not tl.is_file(deref(fname)): return False
    if epochs.size() == 0 and kfolds.size() == 0: return True

    cdef bool found_k = False
    cdef bool found_e = False
    for ix in deref(kfolds):
        tmp = string(b"/") + pfx + ix + string(b"/")
        if not tl.has_string(fname, tmp): continue
        if not tl.ends_with(fname, string(b".root")): continue
        found_k = True; break
   
    for ix in deref(epochs):
        tmp = string(b"/epoch-") + ix + string(b"/")
        if not tl.has_string(fname, tmp): continue
        found_e = True; break

    return found_k and found_e


cdef inline string to_format(vector[string]* fname, string prefx):
    tl = tools()
    cdef string key
    cdef int th = 0
    cdef string emx = b""
    for th in prange(fname.size(), nogil = True, num_threads = 12):
        key = fname.at(th)
        if not tl.has_string(&key, prefx): continue
        tl.replace(&key, prefx, emx)
        break
    return key

cdef class MetricTemplate(Tools):
    cdef metric_template* mtx
    cdef public dict root_leaves
    cdef public dict root_fx
