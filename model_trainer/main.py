from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.bsm_4tops.event_bsm_4tops import *
from AnalysisG.graphs.graph_bsm_4tops import *

from AnalysisG.models.RecursiveGraphNeuralNetwork import *
from AnalysisG.core.lossfx import OptimizerConfig

from AnalysisG.core.io import IO

from runner.samples_mc16 import samples
import argparse
import yaml

parse = argparse.ArgumentParser(description = "Configuration YAML file")
parse.add_argument("--config", action = "store", dest = "config")
parse = parse.parse_args()

try: parse.config
except: print("missing config file. (use --config)"); exit()

f = open(parse.config, "rb")
data = yaml.load(f, Loader = yaml.CLoader)
try: base = data["base"]
except: print("invalid key: expecting 'base' header"); exit()

mc = base["campaign"]

graph_impl = None
graph = base["graph"]
if mc == "mc16" and graph == "GraphTops":          graph_impl = GraphTops()
if mc == "mc16" and graph == "GraphChildren":      graph_impl = GraphChildren()
if mc == "mc16" and graph == "GraphTruthJets":     graph_impl = GraphTruthJets()
if mc == "mc16" and graph == "GraphTruthJetsNoNu": graph_impl = GraphTruthJetsNoNu()
if mc == "mc16" and graph == "GraphJets":          graph_impl = GraphJets()
if mc == "mc16" and graph == "GraphJetsNoNu":      graph_impl = GraphJetsNoNu()
if mc == "mc16" and graph == "GraphDetectorLep":   graph_impl = GraphDetectorLep()
if mc == "mc16" and graph == "GraphDetector":      graph_impl = GraphDetector()
if graph_impl is None: print("invalid graph implementation"); exit()

event = base["event"]
event_impl = None
if mc == "mc16" and event == "BSM4Tops": event_impl = BSM4Tops()
if event_impl is None: print("invalid event implementation"); exit()

out_path  = base["output-path"]
if not out_path.endswith("/"): out_path += "/"
try: out_path += base["project-name"]
except: pass

ana = Analysis()
ana.OutputPath = out_path

files = {}
kill = True
sampl = samples(base["sample-path"], mc)
for i in base["samples"]:
    iox = IO()
    iox.Files = sampl.sample(i)
    iox.Trees = base["tree"]
    if base["samples"][i] == -1: files[i] = iox.Files
    else: files[i] = iox.Files[:base["samples"][i]]
    if len(iox) == 0: print("No Files found ... skipping"); continue
    for x in files[i]:
        ana.AddSamples(x, i)
        ana.AddEvent(event_impl, i)
        ana.AddGraph(graph_impl, i)
    kill = False
if kill: exit()

try: ana.Threads = base["threads"]
except: pass

try: ana.TrainSize = base["training-size"]
except: pass

try: ana.TrainingDataset = base["training-set"]
except: pass

try: ana.kFolds = base["kfolds"]
except: pass

try: ana.kFold = base["kfold"]
except: pass

try: ana.Epochs = base["epochs"]
except: pass

try: ana.Evaluation = base["evaluation"]
except: pass

try: ana.Validation = base["validation"]
except: pass

try: ana.Training = base["training"]
except: pass

try: ana.ContinueTraining = base["continue-training"]
except: pass

try: ana.Targets = base["plot_targets"]
except: pass

models = [i for i in data if i != "base"]
for m in models:
    model_impl = None
    model = data[m]["model"]
    if model == "RecursiveGraphNeuralNetwork": model_impl = RecursiveGraphNeuralNetwork()
    if model_impl is None: print("invalid model implementation"); exit()
    model_impl.device = data[m]["device"]

    try: model_impl.o_edge = data[m]["o_edge"]
    except: pass

    try: model_impl.o_node = data[m]["o_node"]
    except: pass

    try: model_impl.o_graph = data[m]["o_graph"]
    except: pass

    try: model_impl.i_edge = data[m]["i_edge"]
    except: pass

    try: model_impl.i_node = data[m]["i_node"]
    except: pass

    try: model_impl.i_graph = data[m]["i_graph"]
    except: pass

    flgs = {}
    try: flgs = data[m]["extra-flags"]
    except: pass
    for k in flgs: setattr(model_impl, k, flgs[k])

    params = {}
    try: params = data[m]["optimizer"]
    except: pass

    optim = None
    if len(params): optim = OptimizerConfig()
    for i in params: setattr(optim, i, params[i])
    if optim is not None: ana.AddModel(model_impl, optim, m)

    try: chk_pth = data[m]["inference"]["checkpoint_path"]
    except: continue
    model_impl.checkpoint_path = chk_pth
    ana.AddModelInference(model_impl, m)

ana.Start()
