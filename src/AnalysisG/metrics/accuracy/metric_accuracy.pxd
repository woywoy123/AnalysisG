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
    cdef cppclass collector:
        collector() except+
        void inlet(edata* data) except+
 


cdef class AccuracyMetric(MetricTemplate):
    cdef accuracy_metric* mtr
    cdef collector* mcl
    cdef map[string, map[string, map[string, string]]] str_cache
    cdef public default_plt


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

cdef inline void get_data(AccuracyMetric vl, dict data, dict meta):
    cdef edata* edx = new edata(); 
    edx.kfold = meta[b'kfold'] 
    edx.epoch = meta[b'epoch']
    edx.mrk   = meta[b'model_name']

    edx.ntop_score = vget_val(data, b"ntops"  , b"scores", vl)
    edx.ntop_tru   = iget_val(data, b"ntops"  , b"tru"   , vl)
    edx.ntop_prd   = iget_val(data, b"ntops"  , b"prd"   , vl)
    edx.acc_edge   = dget_val(data, b"average", b"edge"  , vl)
    edx.dsid       = iget_val(data, b""       , b"dsid"  , vl)
    edx.prc_idx    = iget_val(data, b"process", b"idx"   , vl)

    edx.training.push_back(make_pdata(data, b"particle"  , b"training", vl))
    edx.training.push_back(make_pdata(data, b"tops_truth", b"training", vl))
    edx.training.push_back(make_pdata(data, b"tops_nom"  , b"training", vl))
    edx.training.push_back(make_pdata(data, b"tops_upr"  , b"training", vl))
    edx.training.push_back(make_pdata(data, b"tops_pr"   , b"training", vl))
    
    edx.validation.push_back(make_pdata(data, b"particle"  , b"validation", vl))
    edx.validation.push_back(make_pdata(data, b"tops_truth", b"validation", vl))
    edx.validation.push_back(make_pdata(data, b"tops_nom"  , b"validation", vl))
    edx.validation.push_back(make_pdata(data, b"tops_upr"  , b"validation", vl))
    edx.validation.push_back(make_pdata(data, b"tops_pr"   , b"validation", vl))

    edx.evaluation.push_back(make_pdata(data, b"particle"  , b"evaluation", vl))
    edx.evaluation.push_back(make_pdata(data, b"tops_truth", b"evaluation", vl))
    edx.evaluation.push_back(make_pdata(data, b"tops_nom"  , b"evaluation", vl))
    edx.evaluation.push_back(make_pdata(data, b"tops_upr"  , b"evaluation", vl))
    edx.evaluation.push_back(make_pdata(data, b"tops_pr"   , b"evaluation", vl))
    vl.mcl.inlet(edx)

    try: assert len(data) == 0; return
    except AssertionError: pass
    print(data)
    exit()
