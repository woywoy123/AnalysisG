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
try: data = data["training"]
except: print("invalid key: expecting 'training' header"); exit()
mc = data["campaign"]
sampl = samples(data["sample-path"], mc)

graph = data["graph"]
graph_impl = None
if mc == "mc16" and graph == "GraphTops":          graph_impl = GraphTops()
if mc == "mc16" and graph == "GraphChildren":      graph_impl = GraphChildren()
if mc == "mc16" and graph == "GraphTruthJets":     graph_impl = GraphTruthJets()
if mc == "mc16" and graph == "GraphTruthJetsNoNu": graph_impl = GraphTruthJetsNoNu()
if mc == "mc16" and graph == "GraphJets":          graph_impl = GraphJets()
if mc == "mc16" and graph == "GraphJetsNoNu":      graph_impl = GraphJetsNoNu()
if graph_impl is None: print("invalid graph implementation"); exit()

event = data["event"]
event_impl = None
if mc == "mc16" and event == "BSM4Tops": event_impl = BSM4Tops()
if event_impl is None: print("invalid event implementation"); exit()

model = data["model"]["name"]
model_impl = None
if model == "RecursiveGraphNeuralNetwork": model_impl = RecursiveGraphNeuralNetwork()
if model_impl is None: print("invalid model implementation"); exit()
model_impl.device = data["model"]["device"]

try: model_impl.o_edge = data["model"]["o_edge"]
except: pass

try: model_impl.o_node = data["model"]["o_node"]
except: pass

try: model_impl.o_graph = data["model"]["o_graph"]
except: pass

try: model_impl.i_edge = data["model"]["i_edge"]
except: pass

try: model_impl.i_node = data["model"]["i_node"]
except: pass

try: model_impl.i_graph = data["model"]["i_graph"]
except: pass

out_path  = data["io"]["output-path"]
try: out_path += data["io"]["project-name"]
except: pass

ana = Analysis()
ana.OutputPath = out_path

train = None
try: train = data["train"]
except: pass

if train is not None:
    optim = OptimizerConfig()
    params = data["optimizer"]
    for i in params: setattr(optim, i, params[i])

    ana.kFolds = data["train"]["kfolds"]
    ana.kFold = data["train"]["kfold"]
    ana.Epochs = data["train"]["epochs"]
    ana.Evaluation = data["train"]["evaluation"]
    ana.Validation = data["train"]["validation"]
    ana.Training = data["train"]["training"]
    ana.ContinueTraining = data["train"]["continue-training"]

    try: ana.TrainSize = data["train"]["training-size"]
    except: pass

    try: ana.TrainingDataset = data["train"]["training-set"]
    except: pass

    ana.AddModel(model_impl, optim, data["run-name"])
else:
    model_impl.checkpoint_path = data["inference"]["checkpoint_path"]
    ana.AddModelInference(model_impl, data["run-name"])

files = {}
for i in data["samples"]:
    iox = IO()
    iox.Files = sampl.sample(i)
    iox.Trees = data["tree"]
    if data["samples"] == -1: files[i] = iox.Files
    else: files[i] = iox.Files[:data["samples"][i]]
    if len(iox) == 0: del files[i]

for i in files:
    for x in files[i]:
        ana.AddSamples(x, i)
        ana.AddEvent(event_impl, i)
        ana.AddGraph(graph_impl, i)

try: ana.Threads = data["threads"]
except: pass

ana.Start()
