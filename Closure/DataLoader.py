import torch
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Event.EventGraph import EventGraphTemplate
from Functions.Event.DataLoader import GenerateDataLoader
import Functions.FeatureTemplates.GraphFeatures as GF
import Functions.FeatureTemplates.EdgeFeatures as EF
import Functions.FeatureTemplates.NodeFeatures as NF

def TrivialNodeFeature(a):
    return float(1.0)

def TrivialEdgeFeature(a, b):
    return float(a.Index*b.Index)

def TrivialGraphFeature(event):
    return float(1.0)


def AttestationIdentity(a, b):
    a = a.tolist()
    b = b.tolist()
    for k, j in zip(a, b):
        try:
            assert k == j
        except: 
            print("-> ", k, j)


def TestEventGraph(Name, Level):
    ev = UnpickleObject(Name)
    for i in ev.Events:
        event = ev.Events[i]
        Particles = getattr(event["nominal"], Level)
        
        eventG = EventGraphTemplate()
        eventG.Event = event["nominal"]
        eventG.Particles = Particles
        eventG.iter = i 
        eventG.SetNodeAttribute("Node", TrivialNodeFeature)
        eventG.SetEdgeAttribute("Edge", TrivialEdgeFeature)      
        eventG.SetGraphAttribute("Graph", TrivialGraphFeature)
        

        # Test With Self Loop
        eventG.SelfLoop = True
        data = eventG.ConvertToData()
        
        AttestationIdentity(data.G_Graph, torch.tensor([[1.]]))
        AttestationIdentity(data.N_Node, torch.tensor([[TrivialNodeFeature(i)] for i in Particles], dtype = torch.float))
        AttestationIdentity(data.E_Edge, torch.tensor([[TrivialEdgeFeature(i, j)] for i in Particles for j in Particles], dtype = torch.float))


        # Test Without Self Loop
        eventG.SelfLoop = False
        data = eventG.ConvertToData()
        AttestationIdentity(data.G_Graph, torch.tensor([[1.]]))
        AttestationIdentity(data.N_Node, torch.tensor([[TrivialNodeFeature(i)] for i in Particles], dtype = torch.float))
        AttestationIdentity(data.E_Edge, torch.tensor([[TrivialEdgeFeature(i, j)] for i in Particles for j in Particles if i != j], dtype = torch.float))
    return True

def TestDataLoader(Name, Level):
    ev = UnpickleObject(Name)
    DL = GenerateDataLoader() 
    DL.CleanUp = False
    DL.AddGraphFeature("Graph", TrivialGraphFeature)
    DL.AddNodeFeature("Node", TrivialNodeFeature)
    DL.AddEdgeFeature("Edge", TrivialEdgeFeature)
    DL.AddSample(ev, "nominal", Level, True)

    for i in DL.DataContainer:
        Particles = getattr(ev.Events[i]["nominal"], Level) 
            
        # Test With Self Loop
        data = DL.DataContainer[i]
        AttestationIdentity(data.G_Graph, torch.tensor([[1.]]))
        AttestationIdentity(data.N_Node, torch.tensor([[TrivialNodeFeature(i)] for i in Particles], dtype = torch.float))
        AttestationIdentity(data.E_Edge, torch.tensor([[TrivialEdgeFeature(i, j)] for i in Particles for j in Particles], dtype = torch.float))
   
    DL = GenerateDataLoader() 
    DL.CleanUp = False
    DL.AddGraphFeature("Graph", TrivialGraphFeature)
    DL.AddNodeFeature("Node", TrivialNodeFeature)
    DL.AddEdgeFeature("Edge", TrivialEdgeFeature)
    DL.AddSample(ev, "nominal", Level, False)
    for i in DL.DataContainer:
        Particles = getattr(ev.Events[i]["nominal"], Level) 

        # Test Without Self Loop
        data = DL.DataContainer[i]
        AttestationIdentity(data.G_Graph, torch.tensor([[1.]]))
        AttestationIdentity(data.N_Node, torch.tensor([[TrivialNodeFeature(i)] for i in Particles], dtype = torch.float))
        AttestationIdentity(data.E_Edge, torch.tensor([[TrivialEdgeFeature(i, j)] for i in Particles for j in Particles if i != j], dtype = torch.float))
    
    DL.MakeTrainingSample(60)
    All = len(list(DL.DataContainer))
    Training_Size = len([k for i in DL.TrainingSample for k in DL.TrainingSample[i]])
    Validation_Size = len([k for i in DL.ValidationSample for k in DL.ValidationSample[i]])
   
    assert float(Training_Size/All)*100 == 60.
    assert float(Validation_Size/All)*100 == 40.


    print("Training Size: " + str(float(Training_Size/All)*100) + " Validation Size: " + str(float(Validation_Size/All)*100))
    return True

def TestDataLoaderMixing(Files, Level):
    
    Loaders = []
    DL = GenerateDataLoader()
    DL.CleanUp = False
    DL.AddGraphFeature("Graph", TrivialGraphFeature)
    DL.AddNodeFeature("Node", TrivialNodeFeature)
    DL.AddEdgeFeature("Edge", TrivialEdgeFeature)
    DL.SetDevice("cuda:0")
    
    Start = []
    End = []
    Samples = []
    su = 0
    for i in Files:
        ev = UnpickleObject(i)
        Loaders.append(ev)
        DL.AddSample(ev, "nominal", Level, True)
        for k in ev.Events:
            if ev.EventIndexFileLookup(k) not in Samples:
                s = ev.EventIndexFileLookup(k)
                Samples.append(s)
                Start.append(su)
                End.append(su + ev.FileEventIndex[s][1] - ev.FileEventIndex[s][0])
            su+=1
     
    assert DL.FileTraces["Start"] == Start
    assert DL.FileTraces["End"] == End 
    assert DL.FileTraces["Samples"] == Samples

    return True



