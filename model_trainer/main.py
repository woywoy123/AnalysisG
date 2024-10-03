from AnalysisG.generators.analysis import Analysis

from AnalysisG.graphs import bsm_4tops, ssml_mc20
from AnalysisG.events.bsm_4tops.event_bsm_4tops import *
from AnalysisG.events.ssml_mc20.event_ssml_mc20 import *

from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.models import *

from AnalysisG.core.io import IO

from runner import samples_mc16, samples_mc20
import argparse
import yaml
import time

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
if mc == "mc16" and graph == "GraphTops":          graph_impl = bsm_4tops.GraphTops()
if mc == "mc16" and graph == "GraphChildren":      graph_impl = bsm_4tops.GraphChildren()
if mc == "mc16" and graph == "GraphTruthJets":     graph_impl = bsm_4tops.GraphTruthJets()
if mc == "mc16" and graph == "GraphTruthJetsNoNu": graph_impl = bsm_4tops.GraphTruthJetsNoNu()
if mc == "mc16" and graph == "GraphJets":          graph_impl = bsm_4tops.GraphJets()
if mc == "mc16" and graph == "GraphJetsNoNu":      graph_impl = bsm_4tops.GraphJetsNoNu()
if mc == "mc16" and graph == "GraphDetectorLep":   graph_impl = bsm_4tops.GraphDetectorLep()
if mc == "mc16" and graph == "GraphDetector":      graph_impl = bsm_4tops.GraphDetector()
if mc == "mc20" and graph == "GraphJets":          graph_impl = ssml_mc20.GraphJets()
if mc == "mc20" and graph == "GraphJetsNoNu":      graph_impl = ssml_mc20.GraphJetsNoNu()
if mc == "mc20" and graph == "GraphDetectorLep":   graph_impl = ssml_mc20.GraphDetectorLep()
if mc == "mc20" and graph == "GraphDetector":      graph_impl = ssml_mc20.GraphDetector()
if graph_impl is None: print("invalid graph implementation"); exit()

event = base["event"]
event_impl = None
if mc == "mc16" and event == "BSM4Tops": event_impl = BSM4Tops()
if mc == "mc20" and event == "SSMLMC":   event_impl = SSML_MC20()
if event_impl is None: print("invalid event implementation"); exit()

out_path  = base["output-path"]
if not out_path.endswith("/"): out_path += "/"
try: out_path += base["project-name"]
except: pass

ana = Analysis()
ana.OutputPath = out_path

files = {}
kill = True
sampl = None
if mc == "mc16": sampl = samples_mc16.samples(base["sample-path"], mc)
if mc == "mc20": sampl = samples_mc20.samples(base["sample-path"], mc)

for i in base["samples"]:
    iox = IO()
    iox.Files = sampl.sample(i)
    iox.Trees = base["tree"]
    if base["samples"][i] == -1: files[i] = iox.Files
    else: files[i] = iox.Files[:base["samples"][i]]
    if len(iox) != 0: continue
    print("No Files found ... skipping")
    try: del files[i]
    except: pass

for x in files:
    for f in files[x]: ana.AddSamples(f, x)
    ana.AddEvent(event_impl, x)
    ana.AddGraph(graph_impl, x)
    kill = False

if kill: exit()

try: ana.Threads = base["threads"]
except: pass

try: ana.BatchSize = base["batch_size"]
except: pass

try: ana.TrainSize = base["training-size"]
except: pass

ana.GraphCache = out_path + "/GraphCache/"
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
    if model == "Experimental": model_impl = Grift()
    if model_impl is None: print("invalid model implementation"); exit()
    model_impl.device = data[m]["device"]

    try: model_impl.o_edge = data[m]["o_edge"]
    except KeyError: pass

    try: model_impl.o_node = data[m]["o_node"]
    except KeyError: pass

    try: model_impl.o_graph = data[m]["o_graph"]
    except KeyError: pass

    try: model_impl.i_edge = data[m]["i_edge"]
    except KeyError: pass

    try: model_impl.i_node = data[m]["i_node"]
    except: pass

    try: model_impl.i_graph = data[m]["i_graph"]
    except KeyError: pass

    flgs = {}
    try: flgs = data[m]["extra-flags"]
    except KeyError: pass
    for k in flgs:
        try: setattr(model_impl, k, flgs[k])
        except: pass

    params = {}
    try: params = data[m]["optimizer"]
    except KeyError: pass

    print("------- Params --------")
    print(yaml.dump(data[m]))
    time.sleep(2)

    optim = None
    if len(params): optim = OptimizerConfig()
    for i in params: setattr(optim, i, params[i])
    if optim is not None: ana.AddModel(model_impl, optim, m)
    data[m]["model"] = [optim, model_impl]

    try: chk_pth = data[m]["inference"]["checkpoint_path"]
    except: continue
    model_impl.checkpoint_path = chk_pth
    ana.AddModelInference(model_impl, m)

ana.Start()
