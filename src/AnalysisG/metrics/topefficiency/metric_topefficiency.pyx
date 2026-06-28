# distutils: language=c++
# cython: language_level=3

cdef class Truth:
    cdef public list truth_mass
    cdef public str process
    def __init__(self, prc):
        self.truth_mass = []
        self.process = env(prc)
        
cdef class Nominal:
    cdef public list kfolds_ntop 
    cdef public list kfolds_chi2
    cdef public list top_mass 

    def __init__(self):
        self.kfolds_ntop = []
        self.kfolds_chi2 = []
        self.top_mass    = []

cdef class Masked:
    cdef public list kfolds_ntop 
    cdef public list kfolds_chi2
    cdef public list kfolds_PR
    cdef public list top_mass

    def __init__(self):
        self.kfolds_ntop = []
        self.kfolds_chi2 = []
        self.kfolds_PR   = []
        self.top_mass    = []


cdef class UnMasked:
    cdef public list kfolds_ntop 
    cdef public list kfolds_chi2
    cdef public list kfolds_PR
    cdef public list top_mass

    def __init__(self):
        self.kfolds_ntop = []
        self.kfolds_chi2 = []
        self.kfolds_PR   = []
        self.top_mass    = []

cdef class Models:
    cdef public dict nominals
    cdef public dict masked 
    cdef public dict unmasked 
    def __init__(self):
        self.nominals = {}
        self.masked   = {}
        self.unmasked = {}

cdef class Epoch:
    cdef public str process
    cdef public Truth truth
    cdef public dict epochs
    def __init__(self, prc):
        self.process = env(prc)
        self.truth  = None
        self.epochs = {}
  

cdef class TopEfficiencyMetric(MetricTemplate):
    def __cinit__(self):
        kins = ["_pt", "_eta", "_phi", "_mass", "_leptonic"]
        xp = [
            ["top_truth"       + k for k in kins], 
            ["top_nominal"     + k for k in kins] + ["top_nominal_chi2"  ], 
            ["top_PR_masked"   + k for k in kins] + ["top_PR_masked_chi2"  , "top_PR_masked_ranks"  ], 
            ["top_PR_unmasked" + k for k in kins] + ["top_PR_unmasked_chi2", "top_PR_unmasked_ranks"], 
        ]
        xp = sum(xp, [])
        self.root_leaves = {"nominal" : xp + ["event_weight"]}
        self.root_fx     = {"nominal" : loader}
        self.mtx = new topefficiency_metric()
        self.mtr = <topefficiency_metric*>(self.mtx)

    def __delloc__(self):
        del self.mtr
        self.mtx = NULL
   
    def Postprocessing(self):
        self.mtr.finalize()
        self.output = {}

        cdef pair[string, map[long, epoch_t]] itr
        cdef pair[string, particle_data_t] itp
        cdef pair[long, epoch_t] ite

        cdef long epc
        cdef str prc, mdl
        
        for itr in self.mtr.generic_data:
            prc = env(itr.first)
            self.output[prc] = Epoch(itr.first)
        
            for ite in self.mtr.generic_data[itr.first]:
                epc = ite.first 
                self.output[prc].epochs[epc] = Models()

                for itp in self.mtr.generic_data[itr.first][ite.first].evn:
                    mdl = env(itp.first)

                    self.output[prc].epochs[epc].nominals[mdl] = Nominal()
                    self.output[prc].epochs[epc].nominals[mdl].kfolds_ntop = itp.second.nominal_kfolds_ntop
                    self.output[prc].epochs[epc].nominals[mdl].kfolds_chi2 = itp.second.nominal_kfolds_chi2
                    self.output[prc].epochs[epc].nominals[mdl].top_mass    = itp.second.nominal_kfolds_top_mass

                    self.output[prc].epochs[epc].unmasked[mdl] = UnMasked()
                    self.output[prc].epochs[epc].unmasked[mdl].kfolds_ntop = itp.second.unmasked_kfolds_ntop
                    self.output[prc].epochs[epc].unmasked[mdl].kfolds_chi2 = itp.second.unmasked_kfolds_chi2
                    self.output[prc].epochs[epc].unmasked[mdl].kfolds_PR   = itp.second.unmasked_kfolds_PR
                    self.output[prc].epochs[epc].unmasked[mdl].top_mass    = itp.second.unmasked_kfolds_top_mass

                    self.output[prc].epochs[epc].masked[mdl] = Masked()
                    self.output[prc].epochs[epc].masked[mdl].kfolds_ntop = itp.second.masked_kfolds_ntop
                    self.output[prc].epochs[epc].masked[mdl].kfolds_chi2 = itp.second.masked_kfolds_chi2
                    self.output[prc].epochs[epc].masked[mdl].kfolds_PR   = itp.second.masked_kfolds_PR
                    self.output[prc].epochs[epc].masked[mdl].top_mass    = itp.second.masked_kfolds_top_mass

                    if self.output[prc].truth is not None: continue
                    self.output[prc].truth = Truth(itr.first)
                    self.output[prc].truth.truth_mass = itp.second.truth_kfolds_top_mass

                    

                
        






        




