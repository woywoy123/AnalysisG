from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.Models import EdgeConv, GCN, InvMassGNN
from Functions.GNN.PathNets import PathNet
from Functions.DataTemplates.DataLoader import GenerateTemplate, ExampleEventGraph, GenerateTemplateCustomSample
from Functions.IO.IO import PickleObject, UnpickleObject

def TestEdgeConvModel():
    
    event1 = ExampleEventGraph()

    Data1 = event1.Data
    Map1 = event1.NodeParticleMap

    Op = Optimizer({}, Debug = True)
    Op.Model = EdgeConv(1, 4)
    Op.sample = Data1
    Op.DefineOptimizer()
    Op.DefineLossFunction("CrossEntropyLoss")
    Op.DefaultTargetType = "Nodes"
    
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
    Op.Model = GCN(1, 4)
    Op.DefineOptimizer()
    Op.DefineLossFunction("CrossEntropyLoss")
    Op.DefaultTargetType = "Nodes"
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
    Op.DefineLossFunction("CrossEntropyLoss")
    Op.DefaultTargetType = "Nodes"
    Op.sample = event1.Data

    P = [event1.NodeParticleMap[i].Index for i in event1.NodeParticleMap]
        
    print("==========")
    for i in range(100):
        Op.TrainClassification()
        _, p = Op.Model(Op.sample).max(1)
        print(p, P, Op.L)

    return True

def TestPathNet():
    
    GenerateTemplate(10)
    events = ExampleEventGraph()
    import torch 
    Op = Optimizer({}, Debug = True)
    Op.DefaultBatchSize = 1
    Op.LearningRate = 1e-3
    Op.WeightDecay = 1e-3
    Op.DefinePathNet(out = 4)
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


def TestJetMergingTagging():
    from Functions.GNN.Models import JetTaggingGNN
    import torch

    #inp = "CustomSignalSample.pkl"
    #f = GenerateTemplateCustomSample(inp, "TruthJetsLep", 20)
    #PickleObject(f, "_Cache/DebugSample.pkl")
    f = UnpickleObject("_Cache/DebugSample.pkl") 
    D = [j for i in f.DataLoader for j in f.DataLoader[i]]

    Op = Optimizer({})
    Op.DefaultBatchSize = 1
    Op.LearningRate = 1e-4
    Op.WeightDecay = 1e-5
    Op.Model = JetTaggingGNN( 1 )
    Op.DefineOptimizer()
    Op.Model.train()
    l = torch.nn.CrossEntropyLoss()
    for x in range(100000):
        for i in D:
            Op.Optimizer.zero_grad()
            Op.sample = i

            p = Op.Model(Op.sample)
            _, x = p.max(1)
            y = Op.sample.y.t().contiguous().squeeze()
            
            #print(x, y, len(x), len(y))
            Op.L = l(p, y)


            Op.L.backward()
            Op.Optimizer.step()
            Op.Optimizer.zero_grad()
            break

        print(l(p, y))
        print(x, y)
 



    return True

