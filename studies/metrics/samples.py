def path_format(ipt): return "/" + "/".join([k for k in ipt.split("/") if len(k) ])
def pfix(ipt, val):   return path_format(val if ipt is None else str(ipt + "/" + val))
def use_l(v1, v2, c): return v1 if c is None else v2
def clk(v1, l = -1): return v1.split("/")[l]
def sani(v1):        return v1.replace("\n", "").replace(" ", "")
def marg(v1, l = 9, right = True): return str(v1) * (not right) + "".join([" " for i in range(l - len(str(v1)))]) + str(v1) * (right)

class DAOD:
    def __init__(self, properties, dlink = None):
        self.dlink= dlink
        self.logical = clk(properties["directory"])
        self.pstcut  = int(properties["Post-Cut"])
        self.precut  = int(properties["Pre-Cut"])
        self.eval  = False
        self.train = False
    

    def __hash__(self): return hash(self.logical)
    def __eq__(self, other): 
        if not isinstance(other, DAOD): return False
        return self.logical == other.logical

    def __add__(self, other):
        if not isinstance(other, DAOD) or self != other: return self
        dl = DAOD({"directory": self.logical}, self.dlink)
        dl.pstcut = self.pstcut + other.pstcut
        dl.precut = self.precut + other.precut
        return dl

    def __str__(self):
        o  = " Post-Cut: " + str(self.pstcut) 
        o += " Pre-Cut: " + str(self.precut) 
        o += " logical: " + str(self.logical)
        if self.dlink: return self.dlink.name + " -> " + o
        return o

class Dataset: 
    def __init__(self, properties, plink = None):
        self.plink    = plink
        self.symbolic = properties["process"]
        self.xsection = float(properties["x-section"])
        self.name     = sani(properties["DatasetName"])
        self.daods    = {}

    def add(self, properties):
        dx = DAOD(properties, self)
        if dx.logical in self.daods: return 
        self.daods[dx.logical] = dx

    def __hash__(self): return hash(self.name)

    def __eq__(self, other): 
        if not isinstance(other, Dataset): return False
        return self.symbolic == other.symbolic

    def __add__(self, other):
        if not isinstance(other, Dataset) or self != other: return self
        dl = Dataset({"x-section": self.xsection, "process": self.symbolic, "DatasetName": self.name}, self.plink)
        dl.daods = self.daods
        for dx in other.daods.values():
            if dx.logical in dl.daods: continue
            dl.daods[dx.logical] = dx
        return dl

class Process:
    def __init__(self, symbolic):
        self.symbolic = symbolic
        self.datasets = {}

    def add(self, properties):
        ds_name = sani(properties["DatasetName"])
        if ds_name not in self.datasets: self.datasets[ds_name] = Dataset(properties, self)
        self.datasets[ds_name].add(properties)

    @property
    def daods(self):
        return {daod.logical: daod for ds in self.datasets.values() for daod in ds.daods.values()}

class Statistics:
    def __init__(self, base_dir=None):
        self.processes = {}
        self.daods     = {}
        self.train     = []
        self.eval      = []

        self.kfolds    = {}
        self.models    = {}

    def __str__(self):
        summary = {}
        for mc in self.processes: 
            for prc in self.processes[mc]:
                prc = self.processes[mc][prc]
                if mc not in summary: summary[mc] = {}
                if prc not in summary[mc]: summary[mc][prc] = {"pre" : 0, "post" : 0, "n-samples" : 0}
                for k in prc.daods: 
                    summary[mc][prc]["post"] += prc.daods[k].pstcut
                    summary[mc][prc]["pre"]  += prc.daods[k].precut
                    summary[mc][prc]["n-samples"] += 1

        out = "======================= SAMPLE DATA ========================="
        for i in summary:
            out += "--------------- campaign: " + str(i) + " -----------------------\n"
            for k in summary[i]:
                out += marg(k.symbolic, right = False) + " " 
                out += "pre-cut: " + marg(summary[i][k]["pre"]) + " "
                out += "post-cut: " + marg(summary[i][k]["post"]) + " "
                out += "n-sampls: " + marg(summary[i][k]["n-samples"], 6) + "\n" 
 

        out += "\n\n\n"
        out += "======================= Training ========================="
        for i in self.models:
            out += "model: " + str(i) + " \n"
            for j in sorted(self.models[i]):
                out += "epoch: " + marg(j, 3, False) + " "
                out += "kfolds: " + str(sorted(list(self.models[i][j]))) + "\n"
        self.summary = summary
        return out




