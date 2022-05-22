from Functions.GNN.Graphs import GenerateDataLoader
from Functions.IO.IO import UnpickleObject
import Functions.DataTemplates.NodeFeatures as nf
import Functions.DataTemplates.EdgeFeatures as ef
import Functions.DataTemplates.GraphFeatures as gf

def GenerateTemplate(Directory = "SignalSample.pkl", Num_events = 1):
    ev = UnpickleObject(Directory)
    Loader = GenerateDataLoader()
    #Loader.Device_s = "cpu"
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddEdgeFeature("dr", d_r)
    Loader.AddEdgeFeature("m", mass) 
    Loader.AddNodeTruth("y", Signal)

    Loader.AddSample(ev, "nominal", "TruthChildren_init")
    
    L = {}
    for i in Loader.EventData:
        ev = Loader.EventData[i]
        it = 0
        L[i] = []
        for k in ev:
            if Num_events == 1:
                PickleObject(k, "Nodes_" + str(i) + ".pkl")
                break
            elif Num_events != 1:
                L[i].append(k)
            
            it += 1
            if it == Num_events:
                PickleObject(L[i], "Nodes_" + str(i) + ".pkl")
                break

def GenerateTemplateCustomSample(Directory, ParticleLevel, Num_events = 1):
    Loader = GenerateDataLoader()
    Loader.NEvents = Num_events

    # ==== Node Features
    Loader.AddNodeFeature("e", nf.energy)
    Loader.AddNodeFeature("eta", nf.eta)
    Loader.AddNodeFeature("phi", nf.phi)
    Loader.AddNodeFeature("pt", nf.pt)
    Loader.AddNodeFeature("m", nf.Mass)

    # ==== Edge Features
    
    # ==== Graph Features
    Loader.AddGraphFeature("met", gf.MissingET)
    Loader.AddGraphFeature("mphi", gf.MissingPhi)
    Loader.AddGraphFeature("MU", gf.MU)
    Loader.AddGraphFeature("NJets", gf.NJets)

    # ==== Node Truth 
    Loader.AddNodeTruth("n_m", nf.Merged)
    
    ev = UnpickleObject(Directory)
    Loader.AddSample(ev, "nominal", ParticleLevel)

    return Loader


def ExampleEventGraph():

    
    GenerateTemplate()
    event = UnpickleObject("Nodes_12.pkl")
    event.SetNodeAttribute("e", energy)
    event.SetNodeAttribute("eta", eta)
    event.SetNodeAttribute("pt", pt)
    event.SetNodeAttribute("phi", phi)
    event.SetEdgeAttribute("dr", d_r)
    event.SetEdgeAttribute("m", mass) 
    event.SetEdgeAttribute("dphi", dphi) 
    event.SetNodeAttribute("y", Signal)
    event.SetNodeAttribute("x", Signal_f)
    event.ConvertToData()


    print("Number of Nodes: ", len(event.Nodes), "Number of Edges: ", len(event.Edges))
    return event
