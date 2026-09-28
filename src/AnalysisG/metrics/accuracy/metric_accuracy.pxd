# distuils: language=c++
# cython: language_level=3

from AnalysisG.core.metric_template cimport *
from cython.operator cimport dereference as deref
from AnalysisG.core.tools cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map    cimport map

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+

cdef extern from "<metrics/samples.h>":
    cdef string mapping(string* fname) except+

cdef extern from "<metrics/transfer.h>":
    cdef cppclass pdata:
        pdata() except+
        vector[double] pt
        vector[double] eta 
        vector[double] phi
        vector[double] energy
        vector[double] mass
        vector[double] score
        vector[int] chn
        string type 
        string mode 

    cdef cppclass edata:
        edata() except+

        vector[pdata*] training    
        vector[pdata*] validation  
        vector[pdata*] evaluation  
        vector[double] ntop_score  

        double acc_edge 

        int prc_idx
        int ntop_tru 
        int ntop_prd 
        int dsid   

        string mrk 
        int kfold 
        int epoch

cdef extern from "<metrics/collector.h>":

    cdef struct evn_t:

        vector[double] weight
        vector[double] masses
        vector[double] error

        vector[int] ntops
        vector[int] ntrus

        vector[vector[double]] mlp
        map[int, map[int, int]] matrix

    cdef struct pairs_t:

        evn_t raw
        evn_t adj

    cdef cppclass performance_t:
         
        map[string, pairs_t] truth
        map[string, pairs_t] nominal
        map[string, pairs_t] unmasked
        map[string, pairs_t] masked

    cdef cppclass performance:

        performance(string mrk_, string mode_) except+
        map[int, map[int, performance_t*]] metric
        string name 
        string mode

    cdef cppclass collector:

        collector() except+
        void inlet(edata* data) except+
        void flush() except+

        map[string, performance*] training
        map[string, performance*] validation
        map[string, performance*] evaluation

        bool release

cdef class Performance:

    cdef performance_t* ptx
    cdef void compile(self)

    cdef bool trig

    cdef string model 
    cdef string mode
    
    cdef map[string, evn_t]* raw_truth
    cdef map[string, evn_t]* raw_nominal
    cdef map[string, evn_t]* raw_unmasked
    cdef map[string, evn_t]* raw_masked

    cdef map[string, evn_t]* adj_truth
    cdef map[string, evn_t]* adj_nominal
    cdef map[string, evn_t]* adj_unmasked
    cdef map[string, evn_t]* adj_masked

    cdef int epoch
    cdef int kfold 

cdef class AccuracyMetric(MetricTemplate):

    cdef accuracy_metric* mtr
    cdef collector* mcl

    cdef map[string, map[string, map[string, string]]] str_cache
    cdef public default_plt

    cdef public dict training 
    cdef public dict validation
    cdef public dict evaluation

cdef inline string m_key(string type_, string mode_, string var_, AccuracyMetric mdl):
    cdef string* val = &mdl.str_cache[type_][mode_][var_]
    if val.size(): return deref(val)
    cdef string kx = type_ + b"_" + var_ if type_.size() else var_
    mdl.str_cache[type_][mode_][var_] = b"accuracy_" + mode_ + b"." + kx + b"." + kx 
    return m_key(type_, mode_, var_, mdl)

cdef inline pdata* make_pdata(dict data, string type_, string mode_, AccuracyMetric mdl):
    cdef pdata* px = new pdata();
    try: px.pt   = data.pop(m_key(type_, mode_, b"pt", mdl))
    except KeyError: del px; return NULL
    px.eta  = data.pop(m_key(type_, mode_, b"eta", mdl))
    px.phi  = data.pop(m_key(type_, mode_, b"phi", mdl))
    try: px.mass   = data.pop(m_key(type_, mode_, b"mass", mdl))
    except KeyError: pass
    try: px.energy = data.pop(m_key(type_, mode_, b"energy", mdl))
    except KeyError: pass
    try: px.score = data.pop(m_key(type_, mode_, b"score", mdl))
    except KeyError: pass
    try: px.chn   = data.pop(m_key(type_, mode_, b"chn", mdl))
    except KeyError: pass
    px.type = type_
    px.mode = mode_
    return px


cdef inline vector[double] vget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    try: return data.pop(m_key(type_, b"training"  , var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"validation", var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"evaluation", var_, mdl))
    except KeyError: pass
    return []

cdef inline int iget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    try: return data.pop(m_key(type_, b"training"  , var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"validation", var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"evaluation", var_, mdl))
    except KeyError: pass
    return 0

cdef inline double dget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    try: return data.pop(m_key(type_, b"training"  , var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"validation", var_, mdl))
    except KeyError: pass
    try: return data.pop(m_key(type_, b"evaluation", var_, mdl))
    except KeyError: pass
    return 0

cdef inline map[string, evn_t]* to_raw(map[string, pairs_t]* ipt):
    cdef map[string, evn_t]* rc = new map[string, evn_t]()
    cdef pair[string, pairs_t] itx
    for itx in deref(ipt): deref(rc)[itx.first] = itx.second.raw
    return rc

cdef inline map[string, evn_t]* to_adj(map[string, pairs_t]* ipt):
    cdef map[string, evn_t]* rc = new map[string, evn_t]()
    cdef pair[string, pairs_t] itx
    for itx in deref(ipt): deref(rc)[itx.first] = itx.second.adj
    return rc

cdef inline dict make_prf(map[string, performance*]* tx, string mode):
    cdef string mdln
    cdef Performance prf
    cdef pair[string, performance*] itr 

    cdef pair[int, performance_t*] itk
    cdef pair[int, map[int, performance_t*]] ite

    cdef dict output = {}
    for itr in deref(tx):
        mdln = itr.first
        for ite in itr.second.metric:
            for itk in ite.second:
                if itk.second == NULL: continue

                prf = Performance()
                prf.model = mdln
                prf.mode  = mode
                prf.epoch = ite.first
                prf.kfold = itk.first
                prf.ptx   = itk.second
                prf.compile()
                itk.second = NULL
               
                try: output[prf.model].append(prf)
                except KeyError: output[prf.model] = [prf]
    return output
