# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp cimport int, float
from libcpp.vector cimport vector
from libcpp.string cimport string

from AnalysisG.core.metric_template cimport *

cdef extern from "<metrics/topefficiency.h>":

    cdef cppclass particle:
        particle() except +

        double pt
        double eta
        double phi
        double mass
        double chi2
        double PR
        int leptonic

    cdef cppclass Event:
        Event() except +

        void make_particle(map[string, vector[double]]* vals) 

        long idx 
        long epoch
        long kfold
        double weight 

        string modelname
        string dset_name 

        vector[particle] top_truth
        vector[particle] top_masked_PR
        vector[particle] top_unmasked_PR
        vector[particle] top_nominal

    cdef cppclass particle_data_t:
        vector[double]  nominal_kfolds_ntop; 
        vector[double]   masked_kfolds_ntop; 
        vector[double] unmasked_kfolds_ntop; 

        vector[double]  nominal_kfolds_chi2; 
        vector[double]   masked_kfolds_chi2; 
        vector[double] unmasked_kfolds_chi2; 


        vector[double]    truth_kfolds_top_mass; 
        vector[double]  nominal_kfolds_top_mass; 
        vector[double]   masked_kfolds_top_mass; 
        vector[double] unmasked_kfolds_top_mass; 

        vector[double]   masked_kfolds_PR; 
        vector[double] unmasked_kfolds_PR; 


    cdef cppclass epoch_t:
        long epoch
        map[string, particle_data_t] evn

    cdef cppclass topefficiency_metric(metric_template):
        topefficiency_metric() except+
        void add_event(Event* ev) except+
        void finalize() except +

        vector[Event*] evnts
        map[string, map[long, epoch_t]] generic_data
  
cdef inline void loader(TopEfficiencyMetric ev, dict data, dict meta):
    cdef string key
    cdef vector[string] dset = tools().split(<string>(meta[b"filename"]), b"/")

    cdef Event* evn = new Event()
    evn.modelname = meta[b"model_name"]
    evn.dset_name = [key for key in dset if tools().has_string(&key, "mc16_13TeV.")][0]
    evn.epoch  = meta[b"epoch"]
    evn.kfold  = meta[b"kfold"]
    evn.idx    = meta[b"index"]
    evn.weight = data.pop(b"nominal.event_weight.event_weight")
    cdef map[string, vector[double]] dt = <map[string, vector[double]]>(data)
    evn.make_particle(&dt)
    ev.mtr.add_event(evn); 

cdef class TopEfficiencyMetric(MetricTemplate):
    cdef topefficiency_metric* mtr
    cdef public dict output
