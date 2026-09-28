from samples import *

class KFolds:
    def __init__(self, base_dir, kgen = None):
        self.base_dir = base_dir
        if kgen is None: return 
        self.folds   = kgen.folds
        self.epoch   = kgen.epoch
        self.modeln  = None
        self.mode    = None
        self.missing = True
        self.ndata   = 0
        self.smpl_l  = None
        self.split   = {}
    
    def model(self, impl):
        self.modeln = impl.name
        return self


    def __hash__(self):
        st  = str(self.epoch) + "-" + str(self.folds)
        st += self.modeln + "-" + self.mode
        return hash(st)

    def __str__(self):
        st  = "model: " + self.modeln + " "
        st += "Mode: "  + self.mode + " "
        st += "epoch: " + str(self.epoch+1) + " "
        st += "kfold: " + str(self.folds+1) + " " 
        try: n = self.ndata
        except AttributeError: n = 0
        st += "Done: " + ("False" if n == 0 else "True")
        return st

    @property
    def _kdir(self): 
        fname  = "epoch-" + str(self.epoch) + "/"
        fname += "kfold-" + str(self.folds) + "_model.pt"
        return fname 

    @property
    def _ddir(self): 
        fname  = self.modeln + "/epoch-" + str(self.epoch)
        fname += "/k-" + str(self.folds) + "/"
        return pfix(self.base_dir, fname)

    def combinations(self):
        kx = []
        for ep in range(self.epoch):
            for k in  range(self.folds):
                fx = KFolds(self.base_dir)
                fx.epoch = ep
                fx.folds = k
                fx.modeln = self.modeln
                kx.append(fx)
        return kx

class session:

    def __init__(self, loc):
        spl = loc.split("/") 
        self.epoch = int([i for i in spl if "epoch-" in i][0].replace("epoch-", "").split("_")[0])
        self.kfold = int([i for i in spl if "kfold-" in i][0].replace("kfold-", "").split("_")[0])
        self.path  = loc

class ModelParams:

    def __init__(self, model, cfg):
        self.model      = model
        self.name       = cfg.name
        self.base_cfg   = cfg

        self.variables  = []
        self.i_node     = []
        self.i_graph    = []
        self.o_edge     = {}
        self.o_graph    = {}

        self.epochs = {}
        self.checkpoints = {}

    def add(self, loc):
        ss = session(loc)
        self.checkpoints[loc] = ss
        if ss.epoch not in self.epochs: self.epochs[ss.epoch] = {}
        self.epochs[ss.epoch][ss.kfold] = session
    
    def node(self, prefx, nf):  self.variables += ["::" + prefx + "::node::" + nf]
    def edge(self, prefx, nf):  self.variables += ["::" + prefx + "::edge::" + nf]
    def graph(self, prefx, nf): self.variables += ["::" + prefx + "::graph::" + nf]
    def extra(self, prefx, nf): self.variables += ["::" + prefx + "::extra::" + nf] 

  
