from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.GNN.Graphs import EventGraph, GenerateDataLoader
from Functions.Plotting.Graphs import Graph
from math import factorial 

def DrawGraph(event, Filename, Dir, Attribute = False):
    G_P = Graph(event)
    G_P.CompileGraph()
    G_P.Filename = Filename 
    G_P.SaveFigure(Dir)

def ValidateNodes(Graph, List):
    try: 
        assert Graph.G.number_of_nodes() == len(List)
        return True
    except AssertionError:
        return False

def ValidateEdges(Graph, List):
    p = len(List)
    p_2 = (factorial(p))/(factorial(2) * factorial( p - 2 ))
    if Graph.SelfLoop:
        p_2 = p_2 + len(List)
    else:
        p_2 = p_2 
    try:
        assert Graph.G.number_of_edges() == int(p_2)
    except AssertionError:
        return True

def TestLevels(event, Tree, Dir):
    G = EventGraph(event, Dir, Tree)
    G.CreateParticleNodes()

    if ValidateNodes(G, G.Particles) == False:
        return False
    DrawGraph(G, "NotConnected", "Plots/EventGraphs/" + Dir)
    
    G.CreateEdges()
    if ValidateEdges(G, G.Particles) == False:
        return False
    DrawGraph(G, "ConnectedSelfLoop", "Plots/EventGraphs/" + Dir)
    
    G = EventGraph(event, Dir, Tree)
    G.SelfLoop = False
    G.CreateParticleNodes()
    G.CreateEdges()

    if ValidateNodes(G, G.Particles) == False or ValidateEdges(G, G.Particles) == False:
        return False
    DrawGraph(G, "Connected", "Plots/EventGraphs/" + Dir)  
    return True


def TestEventGraphs():
    tttt = UnpickleObject("SignalSample.pkl")
    
    for i in tttt.Events:
        ev = tttt.Events[i]

        if TestLevels(ev, "nominal", "TruthTops") == False:
            return False
        if TestLevels(ev, "nominal", "TruthChildren") == False:
            return False
        if TestLevels(ev, "nominal", "TruthChildren_init") == False:
            return False
        if TestLevels(ev, "nominal", "RCJetLepton") == False:
            return False
        if TestLevels(ev, "nominal", "JetLepton") == False:
            return False

        if int(i) == 10:
            break

    ttbar = UnpickleObject("ttbar.pkl")
    
    for i in tttt.Events:
        ev = tttt.Events[i]

        if TestLevels(ev, "nominal", "RCJetLepton") == False:
            return False
        if TestLevels(ev, "nominal", "JetLepton") == False:
            return False
        
        if int(i) == 10:
            break

    return True

def TestDataLoader():
    print("Continue here...")

    return True
