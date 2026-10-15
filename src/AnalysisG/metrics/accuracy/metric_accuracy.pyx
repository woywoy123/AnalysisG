# cython: language = c++
# cython: language_level = 3

from AnalysisG.core.tools cimport *
from libcpp cimport bool, int

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map    cimport pair, map

from cython.operator cimport dereference as deref
from cython.parallel cimport prange

cdef inline dict make_prf(map[string, performance*]* tx, string mode):
    cdef string mdln
    cdef Performance prf
    cdef pair[string, performance*] itr 

    cdef pair[int, performance_t*] itk
    cdef pair[int, map[int, performance_t*]] ite
    
    cdef dict output = {}
    for itr in deref(tx):
        mdln = itr.first
        itr.second.compiler()
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

cdef inline pdata* make_pdata(dict data, string type_, string mode_, AccuracyMetric mdl):
    cdef pdata* px = new pdata();
    TVec(data, mode_, type_, b"pt"    , mdl, &px.pt    )
    TVec(data, mode_, type_, b"eta"   , mdl, &px.eta   )
    TVec(data, mode_, type_, b"phi"   , mdl, &px.phi   )
    TVec(data, mode_, type_, b"mass"  , mdl, &px.mass  )
    TVec(data, mode_, type_, b"energy", mdl, &px.energy)
    TVec(data, mode_, type_, b"score" , mdl, &px.score )
    IVec(data, mode_, type_, b"chn"   , mdl, &px.chn   )
    if px.chn.size() == 0: del px; return NULL 
    px.type = type_
    px.mode = mode_
    return px


cdef inline void get_data(AccuracyMetric vl, dict data, dict meta):

    cdef edata edx = edata(); 
    edx.kfold = meta[b'kfold'] 
    edx.epoch = meta[b'epoch']
    edx.index = meta[b"index"]
    edx.mrk   = meta[b'model_name']

    edx.ntop_score = vget_val(data, b"ntops"  , b"scores", vl)
    edx.ntop_tru   = iget_val(data, b"ntops"  , b"tru"   , vl)
    edx.ntop_prd   = iget_val(data, b"ntops"  , b"prd"   , vl)
    edx.acc_edge   = dget_val(data, b"average", b"edge"  , vl)
    edx.dsid       = iget_val(data, b""       , b"dsid"  , vl)
    edx.prc_idx    = iget_val(data, b"process", b"idx"   , vl)

    cdef int idx = -1
    cdef pdata* nl = make_pdata(data, b"particle", b"training", vl)
    if nl != NULL: 
        edx.training.push_back(nl)
        edx.training.push_back(make_pdata(data, b"tops_truth", b"training", vl))
        edx.training.push_back(make_pdata(data, b"tops_nom"  , b"training", vl))
        edx.training.push_back(make_pdata(data, b"tops_upr"  , b"training", vl))
        edx.training.push_back(make_pdata(data, b"tops_pr"   , b"training", vl))
        idx = meta[b"last_index"][b"accuracy_training"]
 
    nl = make_pdata(data, b"particle", b"validation", vl)
    if nl != NULL:
        edx.validation.push_back(nl)
        edx.validation.push_back(make_pdata(data, b"tops_truth", b"validation", vl))
        edx.validation.push_back(make_pdata(data, b"tops_nom"  , b"validation", vl))
        edx.validation.push_back(make_pdata(data, b"tops_upr"  , b"validation", vl))
        edx.validation.push_back(make_pdata(data, b"tops_pr"   , b"validation", vl))
        idx = meta[b"last_index"][b"accuracy_validation"]

    nl = make_pdata(data, b"particle", b"evaluation", vl)
    if nl != NULL: 
        edx.evaluation.push_back(nl)
        edx.evaluation.push_back(make_pdata(data, b"tops_truth", b"evaluation", vl))
        edx.evaluation.push_back(make_pdata(data, b"tops_nom"  , b"evaluation", vl))
        edx.evaluation.push_back(make_pdata(data, b"tops_upr"  , b"evaluation", vl))
        edx.evaluation.push_back(make_pdata(data, b"tops_pr"   , b"evaluation", vl))
        idx = meta[b"last_index"][b"accuracy_evaluation"]

    if vl.mcl.inlet(&edx) < idx: return
    cdef dict tr = make_prf(&vl.mcl.training  , b"training"  )
    cdef dict va = make_prf(&vl.mcl.validation, b"validation")
    cdef dict ev = make_prf(&vl.mcl.evaluation, b"evaluation")
    for i in tr: 
        try: vl.training[i] += tr[i]
        except KeyError: vl.training[i] = tr[i]

    for i in va: 
        try: vl.validation[i] += va[i]
        except KeyError: vl.validation[i] = va[i]

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
    def __cinit__(self): 
        self.ptx = NULL

    def __init__(self): 
        self.trig = False

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



cdef class Atomic_t:

    def __cinit__(self): self.ptr = NULL

    def __dealloc__(self): del self.ptr

    def __init__(self): 
        self.scoring = ""

    @property
    def truth(self): return self.ptr.truth

    @property
    def masses(self): return self.ptr.masses

    @property
    def weight(self): return self.ptr.weight

    @property
    def index(self): return self.ptr.index

    @property
    def prc(self): return env(deref(self.ptr.prc))

    @property 
    def mode(self): return env(deref(self.ptr.mode))

    @property 
    def kfold(self): return env(deref(self.ptr.kfold))

    @property 
    def epoch(self): return env(deref(self.ptr.epoch))

    @property 
    def model(self): return env(deref(self.ptr.model))


cdef class EvenT:

    def __init__(self): pass

    def __cinit__(self): pass

    def __dealloc__(self):
        del self.data

    cpdef list expand(self, EvenT cx):

        cdef long tx = 0
        cdef long ty = 0
        cdef long py = 0

        cdef long ntops = 0
        cdef long ntrus = 0


        cdef long vdx = 0
        cdef int l, i
        
        cdef pdax_t* dt
        cdef pdax_t* dc
        cdef atomic_t* da 
        cdef vector[long]* itp
        cdef vector[double]* src
        cdef map[int, map[long, atomic_t*]] dxt 

        for i in range(self.vdata.size()):
            dc = cx.vdata[i]
            if   dc.mode == b"training":   l = 0
            elif dc.mode == b"validation": l = 1
            else: l = 2

            for vdx in range(dc.index.size()):
                tx = dc.index.at(vdx)
                
                if not dxt[l].count(tx):
                    da       = new atomic_t()
                    da.prc   = &dc.prc
                    da.mode  = &dc.mode
                    da.kfold = &dc.kfold
                    da.epoch = &dc.epoch
                    da.model = &dc.model
                    da.index = tx
                    dxt[l][tx] = da

                if dc.masses.size() > vdx: 
                    dxt[l][tx].add_truth(dc.masses.at(vdx))

            dt = self.vdata[i]
            for vdx in range(dt.index.size()):
                tx = dt.index.at(vdx)
                if not dxt[l].count(tx): 
                    da       = new atomic_t()
                    da.prc   = &dt.prc
                    da.mode  = &dt.mode
                    da.kfold = &dt.kfold
                    da.epoch = &dt.epoch
                    da.model = &dt.model
                    da.index = tx 
                    dxt[l][tx] = da
                 
                da = dxt[l][tx]
                if dt.masses.size() > vdx: 
                    da.add_value(dt.masses.at(vdx))
                if dt.weight.size() > vdx: 
                    da.add_weight(dt.weight.at(vdx))

        vlx = 0
        cdef Atomic_t ax
        cdef list out = []
        cdef pair[long, atomic_t*] itr
        for l in range(3):
            for itr in dxt[l]:
                ax     = Atomic_t()
                ax.ptr = itr.second
                itr.second.index = vlx
                out.append(ax)
                vlx += 1
        return out

    def __iter__(self):
        self.idx = 0
        self.data = self.vdata.at(0)
        return self

    def __next__(self): 
        if self.idx >= self.vdata.size(): raise StopIteration 
        self.data = self.vdata.at(self.idx)
        self.idx+=1
        return self

    def __len__(self): return self.vdata.size()

    @property
    def mlp(self): return deref(self.data.mlp)

    @property
    def weight(self): return deref(self.data.weight)

    @property
    def masses(self): return deref(self.data.masses)

    @property
    def ntops(self): return deref(self.data.ntops)

    @property
    def ntrus(self): return deref(self.data.ntrus)

    @property 
    def mode(self): return env(self.data.mode)

    @property 
    def prc(self):  return env(self.data.prc)

    @property 
    def kfold(self): return env(self.data.kfold)

    @property 
    def epoch(self): return env(self.data.epoch)

    @property 
    def model(self): return env(self.data.model)





cdef class PDax(Tools):

    def __dealloc__(self): 
        del self.ptx
 
    def __cinit__(self):
        self.ptx = new pdax_t()

    def __init__(self, pkl, fdx): 
        self.cache = {}
        for i in pkl: self.cache[i.replace(".pkl", "")] = {}
        self.idx = fdx; self.jdx = 0 
  
    cdef pdax_t* make(self, dict am, string prc):
        cdef pdax_t* ix = new pdax_t()
        ix.mlp    = ddcast(am.pop("mlp"))
        ix.weight = dcast(am.pop("weight"))
        ix.masses = dcast(am.pop("masses"))
        ix.ntops  = icast(am.pop("ntops"))
        ix.ntrus  = icast(am.pop("ntrus"))
        ix.index  = lcast(am.pop("index"))

        ix.prc    = prc 
        ix.mode   = self.ptx.mode
        ix.epoch  = self.ptx.epoch
        ix.kfold  = self.ptx.kfold
        ix.model  = self.ptx.model
        return ix

    cpdef void Dcx(
        self, dict ax, dict ay, string prc, 
        string raw, string adj
    ):
        cdef pdax_t* ix = self.make(ax, prc); 
        cdef pdax_t* iy = self.make(ay, prc); 
        ix.dmake(iy); 

        self.raw[raw].push_back(ix); 
        self.adj[adj].push_back(iy); 

    cpdef bool load(self, str ax, str ay):
        cdef string raw = enc(ax.replace("Raw", ""))
        cdef string adj = enc(ay.replace("Adj", ""))
        cdef string k 
        cdef list prcs, fxc
        cdef dict _ax, _ay

        for l in self.cache:
            if not self.is_file(l + ".pkl"): continue
            a = l.split("/")
            self.kfold = a[-1]; self.mode  = a[-3]
            self.epoch = a[-2]; self.model = a[-4]
            fxc = self.get_pkl(l)
            if not fxc[0]: continue
            if not len(self.cache[l]): self.cache[l] = fxc[1].__dict__ 
            _ax, _ay = self.cache[l].pop(ax), self.cache[l].pop(ay)
            prcs = list( set( list(_ax) + list(_ay) ))
            for k in prcs: self.Dcx(_ax.pop(k), _ay.pop(k), k, raw, adj)

        return True

    cpdef list data(self):
        cdef pair[string, vector[pdax_t*]] itr
        cdef dict dx = {env(itr.first) : {} for itr in self.adj}
        if not len(dx): return []
        cdef EvenT idx
        for itr in self.adj:
            idx = EvenT()
            idx.idx = 0
            idx.vdata = itr.second
            dx[env(itr.first)]["adj"] = idx 
            if idx.idx > 0: continue
            idx.data = itr.second.at(0)

        for itr in self.raw:
            idx = EvenT()
            idx.idx = 0
            idx.vdata = itr.second
            dx[env(itr.first)]["raw"] = idx
            if idx.idx > 0: continue
            idx.data = itr.second.at(0)
        return [dx]

    @property
    def mode(self): return env(self.ptx.mode)
    @mode.setter
    def mode(self, str val): self.ptx.mode = enc(val)

    @property
    def epoch(self): return env(self.ptx.epoch)
    @epoch.setter
    def epoch(self, str val): self.ptx.epoch = enc(val)

    @property
    def kfold(self): return env(self.ptx.kfold)
    @kfold.setter
    def kfold(self, str val): self.ptx.kfold = enc(val)

    @property
    def model(self): return env(self.ptx.model)
    @model.setter
    def model(self, str val): self.ptx.model = enc(val)





cdef class AccuracyMetric(MetricTemplate):
    def __cinit__(self):
        self.mtx = new accuracy_metric()
        self.mtr = <accuracy_metric*>(self.mtx)
        self.mcl = new collector() 
        self.root_leaves = {}
        self.root_fx = {}
        self.training = {}
        self.validation = {}
        self.evaluation = {}
        self.default_plt = None

    def __init__(self, training, validation, evaluation):
        cdef list ev_ptr  = ["particle_"   + i for i in ["pt", "eta", "phi", "energy", "chn"]]
        cdef list tps_tru = ["tops_truth_" + i for i in ["pt", "eta", "phi", "mass"  , "chn"]]
        cdef list tps_npr = ["tops_pr_"    + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_upr = ["tops_upr_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list tps_nom = ["tops_nom_"   + i for i in ["pt", "eta", "phi", "mass"  , "chn", "score"]]
        cdef list evnt_dt = ["process_idx", "average_edge", "ntops_scores", "dsid", "ntops_tru", "ntops_prd"]
        
        if training:
            self.root_leaves["accuracy_training"]   = ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt
            self.root_fx["accuracy_training"]   = get_data
        if validation: 
            self.root_leaves["accuracy_validation"] = ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt
            self.root_fx["accuracy_validation"] = get_data
        if evaluation: 
            self.root_leaves["accuracy_evaluation"] = ev_ptr + tps_npr + tps_upr + tps_nom + tps_tru + evnt_dt
            self.root_fx["accuracy_evaluation"] = get_data

    def __delloc__(self):
        self.mtx = NULL; 
        del self.mtr
        del self.mcl

    def Postprocessing(self):
        pass


