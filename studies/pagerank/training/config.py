from AnalysisG import Analysis
from AnalysisG.core.lossfx import OptimizerConfig

from AnalysisG.models import *
from AnalysisG.graphs.bsm_4tops.graph_bsm_4tops import *
from AnalysisG.events.bsm_4tops.event_bsm_4tops import *
import yaml
import os

def SwitchType(stw, v1, v2 = True): return v1 if stw else v2

def SetEnvironment(ana, obj, key):
    v = getattr(obj, key)
    if v is None: return 
    setattr(ana, key, v)

def dataset_method(paths):
    dsets = {}
    for subdir, dirs, files in os.walk(paths):
        dset = subdir.split("/")[-1]
        if dset not in dsets: dsets[dset] = []
        files =  [subdir + "/" + i for i in files if i.endswith(".root")]
        if not len(files): continue
        dsets[dset] += files
    return dsets

class ModelYaml:

    def __init__(self, data, name, trace):
        self.yaml = data
        self.yaml["base"] = {}
        self.yaml["optimizer"] = {"parameters" : {}}
        self.yaml["scheduler"] = {"parameters" : {}}
        self.yaml["features"]  = {"graph" : {}, "edge" : {}, "node" : {}}
        self.trace = trace
        self.name = name

    def base(self, model, device):
        self.yaml["base"]["name"]  = self.name
        self.yaml["base"]["model"] = model
        self.yaml["base"]["device"] = device
   
    def _add_feature(self, _type, name, loss, parameters):
        if loss is None: self.yaml["features"][_type][name] = None; return 
        self.yaml["features"][_type][name] = {}
        self.yaml["features"][_type][name][loss] = {}
        for i in parameters: self.yaml["features"][_type][name][loss][i] = parameters[i]

    def optimizer(self, algorithm, parameters):
        self.yaml["optimizer"]["algorithm"] = algorithm
        self.yaml["optimizer"]["parameters"] = {}
        for i in parameters: self.yaml["optimizer"]["parameters"][i] = parameters[i]

    def scheduler(self, algorithm, parameters):
        self.yaml["scheduler"]["algorithm"] = algorithm
        self.yaml["scheduler"]["parameters"] = {}
        for i in parameters: self.yaml["optimizer"]["parameters"][i] = parameters[i]

    def node(self, name, loss = None, parameters = {}): self._add_feature("node", name, loss, parameters)
    def edge(self, name, loss = None, parameters = {}): self._add_feature("edge", name, loss, parameters)
    def graph(self, name, loss = None, parameters = {}): self._add_feature("graph", name, loss, parameters)
    def dump(self): self.trace.data[self.name] = self.yaml

class MatrixCfg:

    def __init__(self, name, outdir, indir, cachedir):
        self.data = yaml.load(open("./training/template.yaml", "rb"), Loader = yaml.CLoader)
        del self.data["model-template"]
        self.data["base"]["models"] = {}
        self.name = name
        self.outdir = outdir
        self.indir  = indir
        self.cachedir = cachedir

    def import_yaml(self, path):
        self.data = yaml.load(open(path, "rb"), Loader = yaml.CLoader)
        return self

    def _assign_keys(self, keys, base, sub):
        for i in keys: self.data[base][sub][i] = keys[i]

    def base(self, keys):
        keys["name"]   = self.name
        keys["output"] = self.outdir 
        keys["input"]  = self.indir + keys["input"]
        self._assign_keys(keys, "base", "project")

    def plotting(self, keys): self._assign_keys(keys, "base", "plotting")
    def event(self, name):    self._assign_keys({"name" : name}, "base", "event") 
    def graph(self, keys):
        keys["path"] = self.cachedir + keys["path"]
        self._assign_keys(keys, "base", "graph")

    def training(self, keys):   
        keys["dataset"] = self.cachedir + keys["dataset"]
        self._assign_keys(keys, "base", "training") 

    def modes(self, keys): self._assign_keys(keys, "base", "modes") 

    def add_model(self, name):
        self.data[name] = {}
        self.data["base"]["models"][name] = None
        return ModelYaml(self.data[name], name, self)

    def get_value(self, base, subdir, key = None):
        if key is None: val = self.data[base][subdir]
        else: val = self.data[base][subdir][key]
        try: 
            if "<" in val and ">" in val: 
                return None
        except: pass
        return val
        
    def dump(self, index = 0):
        pth = self.outdir + "/" + self.name + "/matrix/"
        os.makedirs(pth, exist_ok = True)
        s = yaml.dump(self.data).encode("utf-8")
        f = open(pth + "config-" + str(index) + ".yaml", "wb")
        f.write(s)
        f.close()


