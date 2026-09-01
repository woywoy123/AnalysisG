class KFolds:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.folds    = None
        self.epochs   = None
        self.split    = {}

    @property
    def _fdir(self): return pfix(self.base_dir, self.path)

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

    
