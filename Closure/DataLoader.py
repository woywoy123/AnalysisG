from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.GNN.Graphs import EventGraph, GenerateDataLoader
from Functions.Plotting.Graphs import Graph
from Functions.GNN.Models import EdgeConv
from math import factorial 

def DrawGraph(event, Filename, Dir, Attribute = False):
    G_P = Graph(event)
    G_P.CompileGraph()
    G_P.Filename = Filename 
    G_P.SaveFigure(Dir)

def ValidateNodes(Graph, List):
    try: 
        assert Graph.G.number_of_nodes() == len(List)
        if hasattr(Graph, "Data"):
            assert Graph.Data.num_nodes == len(List)
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
        if hasattr(Graph, "Data"):
            assert Graph.Data.num_edges == int(len(List)*len(List))
        return True
    except AssertionError:
        return False

def TestLevels(event, Tree, Dir):
    G = EventGraph(event, Dir, Tree)
    G.CreateParticleNodes()

    if ValidateNodes(G, G.Particles) == False:
        return False
    DrawGraph(G, "NotConnected", "Plots/EventGraphs_" + Dir)
    
    G.CreateEdges()
    if ValidateEdges(G, G.Particles) == False:
        return False
    DrawGraph(G, "ConnectedSelfLoop", "Plots/EventGraphs_" + Dir)
    
    G = EventGraph(event, Dir, Tree)
    G.SelfLoop = False
    G.CreateParticleNodes()
    G.CreateEdges()

    if ValidateNodes(G, G.Particles) == False or ValidateEdges(G, G.Particles) == False:
        return False
    DrawGraph(G, "Connected", "Plots/EventGraphs_" + Dir)  
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

    tttt = UnpickleObject("SignalSample.pkl")
    ttbar = UnpickleObject("ttbar.pkl")
    

    Loader = GenerateDataLoader()
    Loader.SelfLoop = True
    Loader.AddSample(tttt, "nominal")
    Loader.AddSample(ttbar, "tree")
    Loader.ToDataLoader()
    
    for i in Loader.DataLoader:
        for k in Loader.DataLoader[i]:
            index = int(k.i)
            ev = Loader.EventMap[index]

            if ValidateEdges(ev, ev.Particles) == False:
                return False
            if ValidateNodes(ev, ev.Particles) == False:
                return False
    return True

def TestDataLoaderTrainingValidationTest():
    
    tttt = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    Loader.TestSize = 50
    Loader.ValidationTrainingSize = 50
    Loader.AddSample(tttt, "nominal")
    Loader.MakeTrainingSample()
    Loader.ToDataLoader()
        
    SampleLoader = []
    for i in Loader.DataLoader:
        l = Loader.DataLoader[i]
        for k in l:
            SampleLoader.append(k)
    TestLoader = [] 
    for i in Loader.TestDataLoader:
        l = Loader.TestDataLoader[i]
        for k in l:
            TestLoader.append(k)
    
    if int(len(SampleLoader)/len(TestLoader)) != 0:
        return False



    return True

def TestEventNodeEdgeFeatures():

    def TesterBlock(TruthLevel, tttt):
        
        def Truth(a):
            return a.Signal
        def Index(a):
            return a.Index
        def PT(a):
            return a.pt
        def dPT(a, b):
            return a.pt - b.pt


        tttt_truth = EventGraph(tttt, TruthLevel, "nominal")
        tttt_truth.SelfLoop = True
        tttt_truth.CreateParticleNodes()
        tttt_truth.CreateEdges()
        P = tttt_truth.Particles 
        
        tttt_truth.SetNodeAttribute("y", Truth)
        tttt_truth.SetNodeAttribute("x", Index)
        tttt_truth.SetNodeAttribute("x", PT)
        tttt_truth.SetEdgeAttribute("edge_attr", dPT)
        tttt_truth.ConvertToData()
        GR = tttt_truth.Data
        t = GR.x
        assert list(t.size()) == [len(P), 2]
        
        t = GR.edge_attr
        assert list(t.size()) == [len(P)*len(P), 1]
        
        EC_tttt = EdgeConv(2, 2)
        EC_tttt(GR)

        tttt_truth = EventGraph(tttt, TruthLevel, "nominal")
        tttt_truth.SelfLoop = False
        tttt_truth.CreateParticleNodes()
        tttt_truth.CreateEdges()
        P = tttt_truth.Particles 
        
        tttt_truth.SetNodeAttribute("y", Truth)
        tttt_truth.SetNodeAttribute("x", Index)
        tttt_truth.SetNodeAttribute("x", PT)
        tttt_truth.SetEdgeAttribute("edge_attr", dPT)
        tttt_truth.ConvertToData()
        GR = tttt_truth.Data
        t = GR.x
        assert list(t.size()) == [len(P), 2]
        
        t = GR.edge_attr
        assert list(t.size()) == [len(P)*len(P) - len(P), 1]
        
        EC_tttt = EdgeConv(2, 2)
        EC_tttt(GR)



    tttt = UnpickleObject("SignalSample.pkl")
    tttt = tttt.Events[1]

    try:
        TesterBlock("TruthTops", tttt)
    except AssertionError:
        return False

    try:
        TesterBlock("TruthChildren_init", tttt)
    except AssertionError:
        return False

    try:
        TesterBlock("TruthChildren", tttt)
    except AssertionError:
        return False

    try:
        TesterBlock("JetLepton", tttt)
    except AssertionError:
        return False

    try:
        TesterBlock("RCJetLepton", tttt)
    except AssertionError:
        return False
    return True
