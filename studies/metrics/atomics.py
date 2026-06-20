class Sessions:
    def __init__(self, train_path):
        self.train_path = train_path
        self.variables  = []
        self.o_edge  = []
        self.o_graph = []
        self.i_node  = []
        self.i_graph = []

    def parse(self, data):
        self.abs = data
        spx = data.lstrip(self.train_path).split("/")
        spx = [i for i in spx if len(i)]
        self.loss  = spx[0]
        self.truth = spx[1]
        self.mdl   = spx[2]
        self.mrk   = spx[3]
        self.epoch = spx[5].split("-")[-1]
        self.kfold = spx[6].split("_")[0].split("-")[-1]

    def print(self):
        o  = "======================================\n"
        o += "LOSS METHOD: "     + self.loss   + "\n"
        o += "TRUTH LEVEL: "     + self.truth  + "\n"
        o += "MODEL NAME: "      + self.mdl    + "\n"
        o += "HYPER-PARAMETER: " + self.mrk    + "\n"
        o += "EPOCH: "           + self.epoch  + "\n"
        o += "KFOLD: "           + self.kfold  + "\n"
        o += "======================================\n"
        return o

class ModelParams:

    def __init__(self):
        self.variables  = []
        self.i_node     = []
        self.i_graph    = []
        self.o_edge     = {}
        self.o_graph    = {}


class ModelEnv:

    def __init__(self, sessions, prm):
        self.prm      = ["::".join(i) for i in prm.variables]
        self.sessions = sessions
        self.matrix   = {}

    def compile(self):
        for i in self.sessions:
            ep = i.epoch
            mk = i.mrk
            if mk not in self.matrix:     self.matrix[mk] = {}
            if ep not in self.matrix[mk]: self.matrix[mk][ep] = {}
            self.matrix[mk][ep][i.kfold] = i
            tag = i.mdl + "-" + i.mrk + "::"
            i.variables += [tag + k for k in self.prm]
            i.tag = i.mdl + "-" + i.mrk + "::epoch-" + i.epoch + "::k-" + i.kfold
