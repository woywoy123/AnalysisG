import torch
import networkx as nx
from torch_geometric.data import Data
from Functions.Tools.Alerting import Notification

class EventGraphTemplate:
    def __init__(self):
        self.Particles = []
        self.SelfLoop = False
        self.FullyConnect = True
        self.EdgeAttr = {}
        self.NodeAttr = {}
        self.GraphAttr = {}
        self.Event = ""
        self.Particles = []
        self.iter = -1
        self.Notification = Notification(True)
        self.Notification.Caller = "EventGraph"

    def CreateParticleNodes(self):
        self.G = nx.Graph()
        self.Nodes = []
        self.NodeParticleMap = {}
        for i in range(len(self.Particles)):
            self.Nodes.append(i)
            self.NodeParticleMap[i] = self.Particles[i]
        self.G.add_nodes_from(self.Nodes)

    def CreateEdges(self):
        self.Edges = []
        for i in self.Nodes:
            for j in self.Nodes:
                if self.SelfLoop == False and i == j:
                    continue 
                if self.FullyConnect == False and i != j:
                    continue
                self.Edges.append([i, j])
        self.G.add_edges_from(self.Edges)

    def ConvertToData(self):
        def ApplyToGraph(Dict, Map2 = None):
            for key in Dict:
                fx = Dict[key]
                attr_v = []
                kl = fx.__code__.co_argcount
                
                if Map2 == None:
                    attr_v += [[fx(self.Event)]]
                    k = "G_"
                elif kl == 1:
                    attr_v += [[ fx(self.NodeParticleMap[n]) ] for n in Map2]
                    k = "N_"
                else:
                    attr_v += [[ fx(self.NodeParticleMap[n[0]], self.NodeParticleMap[n[1]]) ] for n in Map2]
                    k = "E_"
                setattr(self.Data, k + key, torch.tensor(attr_v, dtype = torch.float))
                
        self.CreateParticleNodes()
        self.CreateEdges()
        
        edge_index = torch.tensor(self.Edges, dtype=torch.long).t().contiguous()
        self.Data = Data(edge_index = edge_index)

        ApplyToGraph(self.GraphAttr)
        ApplyToGraph(self.NodeAttr, self.NodeParticleMap)
        ApplyToGraph(self.EdgeAttr, self.Edges)

        self.Data.num_nodes = torch.tensor(len(self.Particles))
        setattr(self.Data, "i", torch.tensor(self.iter))
        return self.Data

    def __SetAttribute(self, c_name, fx, container):
        if c_name not in container:
            container[c_name] = fx
        else:
            self.Notification.Warning("Found Duplicate " + c_name + " Attribute")

    def SetNodeAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.NodeAttr)

    def SetEdgeAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.EdgeAttr)

    def SetGraphAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.GraphAttr)

class EventGraphTruthTops(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TruthTops

class EventGraphTruthTopChildren(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TopPostFSRChildren
    
class EventGraphTruthJetLepton(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TruthJets
        self.Particles += self.Event.Electrons
        self.Particles += self.Event.Muons

class EventGraphDetector(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.DetectorParticles


