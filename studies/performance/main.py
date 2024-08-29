from AnalysisG.generators import Analysis

from AnalysisG.models import *
model_method = {"RecursiveGraphNeuralNetwork" : RecursiveGraphNeuralNetwork, "Experimental" : Experimental}

# event implementation
from AnalysisG.events.gnn import EventGNN
from AnalysisG.events.bsm_4tops import BSM4Tops
event_method = {"BSM4Tops" : BSM4Tops, "EventGNN" : EventGNN}

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

# study
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
import pickle


mass = "None"
study = "topefficiency"
figure_path = "./Output/"

graph_name = "GraphTruthJets"
event_name = "BSM4Tops"
model_name = "Experimental"

root_model = "/CERN/trainings/results/models/"

model_states = {
        "MRK-1" : {
            "epoch-1" : (["kfold-10"], ["cuda:0"])
        }
}

sample_path = {
        "single-top" : "/CERN/trainings/mc16-full/sorted-data/singletop/*"
}

ana = Analysis()
#ana.GraphCache = "ProjectName"
ana.FetchMeta = True

for j in sample_path:
    try: ana.AddSamples(sample_path[j], j)
    except KeyError: continue
    ana.AddEvent(event_method[event_name](), j)
    #except KeyError: continue
    ana.AddGraph(graph_method[graph_name](), j)
    #except KeyError: continue

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
            ana.AddModelInference(gnn, "ROOT/" + graph_name + "_" + model_name + "/" + j + "/" + ep)
ana.Start()






#method = plotting_method[study]
#method.figures.figure_path = figure_path
#method.figures.mass_point  = "Mass." + mass + ".GeV"


#for f in files:
#    pth = f.split("/")[-2]
#    Path("./serialized-data/").mkdir(parents = True, exist_ok = True)
#    if not gen_data: continue
#    ev = EventGNN() if model_ev else BSM4Tops()
#
#    sel = None
#    if study == "topefficiency": sel = TopEfficiency()
#
#    ana = Analysis()
#    ana.AddSamples(f, "tmp")
#    ana.AddEvent(ev, "tmp")
#    ana.AddSelection(sel)
#    ana.Threads = 10
#    ana.Start()
#
#    f = open("./serialized-data/" + pth + ".pkl", "wb")
#    pickle.dump(sel, f)
#    f.close()
#
##f = open(study + "-" + mass_point + ".pkl", "rb")
##pres = pickle.load(f)
##f.close()
#
#pres = "./serialized-data/"
#print("plotting: " + study)
#if study == "topefficiency": method.figures.TopEfficiency(pres)
