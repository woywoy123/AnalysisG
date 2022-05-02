import torch
import networkx as nx
from torch_geometric.data import Data
from Functions.Tools.Alerting import Notification

class EventGraphTemplate:
    def __init__(self):
        self.Particles = []
        self.SelfLoop = False
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
                self.Edges.append([i, j])
        self.G.add_edges_from(self.Edges)

    def ConvertToData(self):
        def ApplyToGraph(Dict, Map2 = []):
            for key in Dict:
                fx = Dict[key]
                attr_v = []
                k = ""
                for n in Map2:
                    if isinstance(n, int):
                        attr_v.append(fx(self.NodeParticleMap[n]))
                        k = "N_"
                    if isinstance(n, list):
                        attr_v.append(fx(self.NodeParticleMap[n[0]], self.NodeParticleMap[n[1]]))
                        k = "E_"
                if len(attr_v) == 0:
                    k = "G_"
                    attr_v = fx(self.Event)
                setattr(self.Data, k + key, torch.tensor(attr_v, dtype = torch.float))

        self.CreateParticleNodes()
        self.CreateEdges()
        
        edge_index = torch.tensor(self.Edges, dtype=torch.long).t().contiguous()
        self.Data = Data(edge_index = edge_index)

        ApplyToGraph(self.GraphAttr)
        ApplyToGraph(self.NodeAttr, self.NodeParticleMap)
        ApplyToGraph(self.EdgeAttr, self.Edges)

        setattr(self.Data, "i", self.iter)  
        self.Data.num_nodes = len(self.Particles)
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


