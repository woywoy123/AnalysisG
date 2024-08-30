from AnalysisG.generators import Analysis
import pickle

# model implementations
from AnalysisG.models import *
model_method = {"RecursiveGraphNeuralNetwork" : RecursiveGraphNeuralNetwork, "Experimental" : Experimental}

# selection studies
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
selection_method = {"TopEfficiency" : TopEfficiency}
selection_container = {}

# event implementations
from AnalysisG.events.gnn import EventGNN
from AnalysisG.events.bsm_4tops import BSM4Tops
event_method = {"BSM4Tops" : BSM4Tops, "EventGNN" : EventGNN}

# graph implementations
from AnalysisG.graphs.bsm_4tops import *
graph_method = {
        "GraphTruthJets"     : GraphTruthJets,
        "GraphTruthJetsNoNu" : GraphTruthJetsNoNu,
        "GraphJets"          : GraphJets,
        "GraphJetsNoNu"      : GraphJetsNoNu,
        "GraphDetectorLep"   : GraphDetectorLep,
        "GraphDetector"      : GraphDetector
}

# figures
import topefficiency
plotting_method = {"topefficiency" : topefficiency}

study = "topefficiency"
figure_path = "./Output/"
graph_name = "GraphJets"
inference_mode = True

root_model = "/CERN/trainings/results/models/"
data_path  = "/CERN/trainings/mc16-full/"

model_states = {}
if inference_mode:
    model_states |= {
            "MRK-1" : {"epoch-74" : (["kfold-5"], ["cuda:0"])}
    }

sample_path = {
        #"single-top" : data_path + "sorted-data/singletop/*",
        #"ttbar"      : data_path + "sorted-data/ttbar/mc16_13TeV.412070.aMcAtNloPy8EG_A14_ttbar_hdamp258p75_dil_BFiltBBVeto.deriv.DAOD_TOPQ1.e7129_a875_r9364_p4514/DAOD_TOPQ1.40945514._000006.root"
        #"single-top" : "ProjectName/ROOT/GraphTruthJets_Experimental/MRK-1/epoch-1/*"
        "ttbar" : "ProjectName/ROOT/GraphJets_Experimental/MRK-1/epoch-74/*"
}

event_name     = "BSM4Tops"      if inference_mode     else "EventGNN"
graph_name     = graph_name      if inference_mode     else ""
model_name     = "Experimental"  if inference_mode     else ""
selection_name = "TopEfficiency" if not inference_mode else ""

ana = Analysis()
ana.Threads = 1
ana.GraphCache = "ProjectName"

if len(model_states): ana.FetchMeta = True
for j in sample_path:
    try: ana.AddSamples(sample_path[j], j)
    except KeyError: pass

    try: ana.AddEvent(event_method[event_name](), j)
    except KeyError: pass
    except NameError: pass

    try: ana.AddGraph(graph_method[graph_name](), j)
    except KeyError: pass
    except NameError: pass

try:
    selection_container[selection_name] = selection_method[selection_name]()
    ana.AddSelection(selection_container[selection_name])
except KeyError: pass
except NameError: pass

for j in model_states:
    for ep in model_states[j]:
        mdl = root_model + graph_name + "/" + model_name + "/" + j + "/state/" + ep + "/"
        kf, dev = model_states[j][ep]
        for i in range(len(kf)):
            gnn = model_method[model_name]()
            gnn.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
            gnn.o_graph = {"ntops"   : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
            gnn.i_node = ["pt", "eta", "phi", "energy"]
            gnn.device = dev[i]
            gnn.checkpoint_path = mdl + kf[i] + "_model.pt"
            ana.AddModelInference(gnn, "ROOT/" + graph_name + "_" + model_name + "/" + j + "/" + ep + "/" + kf[i])
ana.Start()

for j in selection_container:
    f = open("./serialized-data/" + j +".pkl", "wb")
    pickle.dump(selection_container[j], f)
    f.close()


method = plotting_method[study]
method.figures.figure_path = figure_path
if study == "topefficiency": method.figures.TopEfficiency("./serialized-data/")
