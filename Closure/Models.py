from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.Models import EdgeConv, GCN, InvMassGNN, PathNet
from Functions.Plotting.Histograms import TH2F, TH1F
from skhep.math.vectors import LorentzVector

def GenerateTemplate(Num_events = 1):
    def eta(a):
        return float(a.eta)
    
    def energy(a):
        return float(a.e)
    
    def pt(a):
        return float(a.pt)
    
    def phi(a):
        return float(a.phi)
    
    def d_r(a, b):
        return float(a.DeltaR(b))

    def Signal(a):
        return int(a.Index)

    def m(a, b):
        t_i = LorentzVector()
        t_i.setptetaphie(a.pt, a.eta, a.phi, a.e)

        t_j = LorentzVector()
        t_j.setptetaphie(b.pt, b.eta, b.phi, b.e)

        T = t_i + t_j
        return float(T.mass)



    ev = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    #Loader.Device_s = "cpu"
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddEdgeFeature("dr", d_r)
    Loader.AddEdgeFeature("m", m) 
    Loader.AddNodeTruth("y", Signal)

    Loader.AddSample(ev, "nominal", "TruthChildren_init")
    Loader.ToDataLoader()
    
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

def ExampleEventGraph():

    # Different features to include as node and edges
    def eta(a):
        return float(a.eta)
    def energy(a):
        return float(a.e)
    def pt(a):
        return float(a.pt)
    def phi(a):
        return float(a.phi)
    def d_r(a, b):
        return float(a.DeltaR(b))
    def Signal(a):
        return int(a.Index)
    def m(a, b):
        t_i = LorentzVector()
        t_i.setptetaphie(a.pt, a.eta, a.phi, a.e)

        t_j = LorentzVector()
        t_j.setptetaphie(b.pt, b.eta, b.phi, b.e)

        T = t_i + t_j
        return float(T.mass)
    
    GenerateTemplate()
    event = UnpickleObject("Nodes_10.pkl")
    event = event.Data

    event.SetNodeAttribute("e", energy)
    event.SetNodeAttribute("eta", eta)
    event.SetNodeAttribute("pt", pt)
    event.SetNodeAttribute("phi", phi)
    event.SetEdgeAttribute("dr", d_r)
    event.SetEdgeAttribute("m", m) 
    event.SetNodeAttribute("y", Signal)
    event.ConvertToData()


    print("Number of Nodes: ", len(event.Nodes), "Number of Edges: ", len(event.Edges))
    return event


def TestEdgeConvModel():
    
    event1 = ExampleEventGraph()

    Data1 = event1.Data
    Map1 = event1.NodeParticleMap

    Op = Optimizer({}, Debug = True)
    Op.Model = EdgeConv(4, 2)
    Op.DefineOptimizer()
    Op.sample = Data1
    
    for i in range(10):
        Op.TrainClassification()
        print(Op.L)

    return True

def TestGCNModel():
    
    event1 = ExampleEventGraph()
    print("Number of Nodes: ", len(event1.Nodes), "Number of Edges: ", len(event1.Edges))

    Data1 = event1.Data
    Map1 = event1.NodeParticleMap

    Op = Optimizer({}, Debug = True)
    Op.Model = GCN(4, 2)
    Op.DefineOptimizer()
    Op.sample = Data1
    
    for i in range(10):
        Op.TrainClassification()
        print(Op.Model(Data1).max(1))

    return True

def TestInvMassGNN():
    
    event1 = ExampleEventGraph()

    Op = Optimizer({}, Debug = True)
    Op.Model = InvMassGNN(4)
    Op.LearningRate = 1e-5
    Op.WeightDecay = 1e-3
    Op.DefineOptimizer()
    Op.sample = event1.Data

    P = [event1.NodeParticleMap[i].Index for i in event1.NodeParticleMap]
        
    print("==========")
    for i in range(100000):
        Op.TrainClassification()
        _, p = Op.Model(Op.sample).max(1)
        print(p, P, Op.L)

    return True

def TestPathNet():
    
    GenerateTemplate(10)
    events = ExampleEventGraph()
    import torch 
    Op = Optimizer({}, Debug = True)
    Op.LearningRate = 1e-5
    Op.WeightDecay = 1e-3
    Op.DefinePathNet()
    Op.sample = events.Data

    P = [events.NodeParticleMap[i].Index for i in events.NodeParticleMap]
    M_P = [events.NodeParticleMap[i] for i in events.NodeParticleMap]
    from Functions.Particles.Particles import Particle 
    M = [Particle(True) for i in range(4)]
    for i, j in zip(P, M_P):
        M[i].Decay.append(j)
    x = []
    for i in M:
        i.CalculateMassFromChildren()
        x.append(i.Mass_GeV) 
       

    print("==========")
    for i in range(100000):
        Op.TrainClassification()
        _, p = Op.Model(Op.sample).max(1)
       
        print(p, P, Op.L)
        print(x) 
        if p.tolist() == P:
            break
    return True



