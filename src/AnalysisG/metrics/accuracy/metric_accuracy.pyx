# distutils: language=c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from AnalysisG.core.tools cimport enc, env

cdef inline void get_data(AccuracyMetric vl, dict data, dict meta):

    cdef edata edx = edata(); 
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
    vl.mcl.inlet(&edx)

    if not vl.mcl.release: return
    vl.mcl.release = False
    cdef dict tr = make_prf(&vl.mcl.training  , b"training")
    for i in tr: 
        try: vl.training[i] += tr[i]
        except KeyError: vl.training[i] = tr[i]
 
    cdef dict va = make_prf(&vl.mcl.validation, b"validation")
    for i in va: 
        try: vl.validation[i] += va[i]
        except KeyError: vl.validation[i] = va[i]
 
    cdef dict ev = make_prf(&vl.mcl.evaluation, b"evaluation")
    for i in ev: 
        try: vl.evaluation[i] += ev[i]
        except KeyError: vl.evaluation[i] = ev[i]
   
    vl.Postprocessing()
    vl.mcl.flush()
    try: assert len(data) == 0; return
    except AssertionError: pass
    print(data)
    exit()


cdef class Performance:
    def __cinit__(self): self.ptx = NULL
    def __init__(self): self.trig = False
    def __dealloc__(self):
        if self.raw_truth    != NULL: del self.raw_truth   
        if self.adj_truth    != NULL: del self.adj_truth   

        if self.raw_nominal  != NULL: del self.raw_nominal 
        if self.adj_nominal  != NULL: del self.adj_nominal 

        if self.raw_unmasked != NULL: del self.raw_unmasked
        if self.adj_unmasked != NULL: del self.adj_unmasked

        if self.raw_masked   != NULL: del self.raw_masked  
        if self.adj_masked   != NULL: del self.adj_masked 

    def __hash__(self):
        cdef str cfg = env(self.model)
        cfg += env(self.mode)
        cfg += str(self.epoch)
        cfg += str(self.kfold)
        return hash(cfg)

    def __eq__(self, obj):
        return hash(obj) == hash(self)

    cdef void compile(self):
        if self.trig: return
        self.trig = True
        self.raw_truth    = to_raw(&self.ptx.truth)
        self.adj_truth    = to_adj(&self.ptx.truth)
        self.ptx.truth.clear()

        self.raw_nominal  = to_raw(&self.ptx.nominal)
        self.adj_nominal  = to_adj(&self.ptx.nominal)
        self.ptx.nominal.clear()

        self.raw_unmasked = to_raw(&self.ptx.unmasked)
        self.adj_unmasked = to_adj(&self.ptx.unmasked)
        self.ptx.unmasked.clear()

        self.raw_masked   = to_raw(&self.ptx.masked)
        self.adj_masked   = to_adj(&self.ptx.masked)
        self.ptx.masked.clear()

    @property
    def ModelName(self): return env(self.model)
    @property
    def ModeName(self): return env(self.mode)

    @property
    def kFold(self): return self.kfold
    @property
    def Epoch(self): return self.epoch

    @property
    def RawTruth(self): return deref(self.raw_truth)
    @property
    def AdjTruth(self): return deref(self.adj_truth)

    @property
    def RawNominal(self): return deref(self.raw_nominal)
    @property
    def AdjNominal(self): return deref(self.adj_nominal)

    @property
    def RawUnmasked(self): return deref(self.raw_unmasked)
    @property
    def AdjUnmasked(self): return deref(self.adj_unmasked)

    @property
    def RawMasked(self): return deref(self.raw_masked)
    @property
    def AdjMasked(self): return deref(self.adj_masked)



cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):
        cdef list ev_ptr  = ["particle_"   + i for i in ["pt", "eta", "phi", "energy", "chn"]]
        cdef list tps_tru = ["tops_truth_" + i for i in ["pt", "eta", "phi", "mass"  , "chn"]]
        cdef list tps_npr = ["tops_pr_"    + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_upr = ["tops_upr_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_nom = ["tops_nom_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list evnt_dt = ["process_idx", "average_edge", "ntops_scores", "dsid", "ntops_tru", "ntops_prd"]

        self.root_leaves = {
                "accuracy_training"   : ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt,
                #"accuracy_validation" : ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt,
                #"accuracy_evaluation" : ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt
        }

        self.root_fx = {
                "accuracy_training"   : get_data,
                #"accuracy_validation" : get_data,
                #"accuracy_evaluation" : get_data
        }

        self.mtx = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.mtx)
        self.mcl = new collector() 
        
        self.training   = {}
        self.validation = {}
        self.evaluation = {}

        self.default_plt = None

    def __delloc__(self):
        self.mtx = NULL; 
        del self.mtr
        del self.mcl

    def Postprocessing(self):
        pass
