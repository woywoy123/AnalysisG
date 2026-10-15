# distuils: language=c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from AnalysisG.core.metric_template cimport *
from AnalysisG.core.tools cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map    cimport map, pair

cdef extern from "<metrics/samples.h>":
    cdef string mapping(string* fname) except+

cdef extern from "<metrics/accuracy.h>":
    cdef cppclass accuracy_metric(metric_template):
        accuracy_metric() except+


cdef inline vector[double]* dcast(list v):
    return  new vector[double](<vector[double]>(v))

cdef inline vector[vector[double]]* ddcast(list v):
    return  new vector[vector[double]](<vector[vector[double]]>(v))

cdef inline vector[int]* icast(list v):
    return new vector[int](<vector[int]>(v))

cdef inline vector[long]* lcast(list v):
    return new vector[long](<vector[long]>(v))



cdef inline string* scast(str v):
    return new string(enc(v))

cdef extern from "<metrics/transfer.h>":

    cdef cppclass pdata:

        pdata() except+
        vector[double] pt
        vector[double] eta 
        vector[double] phi
        vector[double] energy
        vector[double] mass
        vector[double] score
        vector[long] index
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
        long index

        int prc_idx
        int ntop_tru 
        int ntop_prd 
        int dsid   

        string mrk 
        int kfold 
        int epoch

cdef extern from "<metrics/collector.h>":

    cdef cppclass performance:

        performance(string mrk_, string mode_) except+

        void compiler() except+

        map[int, map[int, performance_t*]] metric
        string name 
        string mode

    cdef cppclass collector:

        collector() except+
        int inlet(edata* data) except+
        void flush() except+

        map[string, performance*] training
        map[string, performance*] validation
        map[string, performance*] evaluation

        bool release

    cdef struct evn_t:

        vector[double] weight
        vector[double] masses
        vector[double] error
        vector[long]   index 

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

    cdef public long epoch
    cdef public long kfold 

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

cdef inline bool TVec(dict data, string mode_, string type_, string var_, AccuracyMetric mdl, vector[double]* F):
    try: F[0] = data.pop(m_key(type_, mode_, var_, mdl))
    except KeyError: return False
    return True

cdef inline bool IVec(dict data, string mode_, string type_, string var_, AccuracyMetric mdl, vector[int]* F):
    try: F[0] = data.pop(m_key(type_, mode_, var_, mdl))
    except KeyError: return False
    return True

cdef inline bool Tint(dict data, string mode_, string type_, string var_, AccuracyMetric mdl, int* F):
    try: F[0] = data.pop(m_key(type_, mode_, var_, mdl))
    except KeyError: return False
    return True

cdef inline double Tdbl(dict data, string mode_, string type_, string var_, AccuracyMetric mdl, double* F):
    try: F[0] = data.pop(m_key(type_, mode_, var_, mdl))
    except KeyError: return False
    return True

cdef inline vector[double] vget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    cdef vector[double] o
    if TVec(data, b"training"  , type_, var_, mdl, &o): return o
    if TVec(data, b"validation", type_, var_, mdl, &o): return o
    if TVec(data, b"evaluation", type_, var_, mdl, &o): return o
    return []

cdef inline int iget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    cdef int o
    if Tint(data, b"training"  , type_, var_, mdl, &o): return o
    if Tint(data, b"validation", type_, var_, mdl, &o): return o
    if Tint(data, b"evaluation", type_, var_, mdl, &o): return o
    return 0

cdef inline double dget_val(dict data, string type_, string var_, AccuracyMetric mdl):
    cdef double o
    if Tdbl(data, b"training"  , type_, var_, mdl, &o): return o
    if Tdbl(data, b"validation", type_, var_, mdl, &o): return o
    if Tdbl(data, b"evaluation", type_, var_, mdl, &o): return o
    return 0

cdef inline map[string, evn_t]* to_raw(map[string, pairs_t]* ipt):
    cdef pair[string, pairs_t] itx
    cdef map[string, evn_t]* rc = new map[string, evn_t]()
    for itx in deref(ipt): deref(rc)[itx.first] = itx.second.raw
    return rc

cdef inline map[string, evn_t]* to_adj(map[string, pairs_t]* ipt):
    cdef pair[string, pairs_t] itx
    cdef map[string, evn_t]* rc = new map[string, evn_t]()
    for itx in deref(ipt): deref(rc)[itx.first] = itx.second.adj
    return rc

cdef extern from "<metrics/atomics.h>":

    cdef cppclass pdax_t(tools):
        
        pdax_t()               except+
        void dmake(pdax_t* vl) except+

        vector[vector[double]]* mlp
        vector[double]* weight 
        vector[double]* masses
        vector[long]* index

        vector[int]* ntops
        vector[int]* ntrus

        string mode 
        string prc  
        string kfold
        string epoch
        string model

    cdef cppclass atomic_t(tools) nogil:

        atomic_t() except+

        long add_truth(double m) except+
        long add_value(double m) except+
        long add_weight(double m) except+

        vector[double] truth
        vector[double] masses
        vector[double] weight
        
        string* prc
        string* mode 
        string* kfold
        string* epoch
        string* model

        long index 


cdef class Atomic_t:

    cdef atomic_t* ptr
    cdef public scoring
   
cdef class EvenT:

    cdef pdax_t* data
    cdef vector[pdax_t*] vdata
    cdef vector[atomic_t*] evnts
    cdef long idx

    cpdef list expand(self, EvenT cx)


cdef class PDax(Tools):
    
    cdef pdax_t* make(self, dict am, string prc)
    cpdef void Dcx(self, dict ar, dict ax, string prc, string raw, string adj)
    cpdef bool load(self, str a1, str a2)
    cpdef list data(self)

    cdef pdax_t* ptx
    cdef map[string, vector[pdax_t*]] adj
    cdef map[string, vector[pdax_t*]] raw

    cdef public dict cache
    cdef public int idx
    cdef public int jdx
    
    cdef public bool is_av
    
    


