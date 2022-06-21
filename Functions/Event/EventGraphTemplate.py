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
        # Anticipates Errors when certain particles dont have an attribute
        def ErrorHandler(fx, *args):
            try: 
                if len(args) == 1:
                    return fx(args[0])
                else:
                    return fx(args[0], args[1])
            except AttributeError:
                return 0

        def ApplyToGraph(Dict, Map2 = None, Preprocess = False):
            for key in Dict:
                if key[:2] != "P_" and Preprocess == True:
                    continue
                elif key[:2] == "P_" and Preprocess == False:
                    continue

                fx = Dict[key]
                attr_v = []
                kl = fx.__code__.co_argcount
                
                if Map2 == None:
                    attr_v += [[ErrorHandler(fx, self.Event)]]
                    k = "G_"
                elif kl == 1:
                    attr_v += [[ ErrorHandler(fx, self.NodeParticleMap[n]) ] for n in Map2]
                    k = "N_"
                else:
                    attr_v += [[ ErrorHandler(fx, self.NodeParticleMap[n[0]], self.NodeParticleMap[n[1]]) ] for n in Map2]
                    k = "E_"
                
                if Preprocess == True:
                    continue
                setattr(self.Data, k + key, torch.tensor(attr_v, dtype = torch.float))
                
        self.CreateParticleNodes()
        self.CreateEdges()
        
        edge_index = torch.tensor(self.Edges, dtype=torch.long).t().contiguous()
        self.Data = Data(edge_index = edge_index)
        
        # Do some preprocessing 
        ApplyToGraph(self.GraphAttr, Preprocess = True)
        ApplyToGraph(self.NodeAttr, self.NodeParticleMap, Preprocess = True)
        ApplyToGraph(self.EdgeAttr, self.Edges, Preprocess = True)
        
        # Apply to graph 
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

    def SetEdgeAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.EdgeAttr)

    def SetNodeAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.NodeAttr)

    def SetGraphAttribute(self, c_name, fx):
        self.__SetAttribute(c_name, fx, self.GraphAttr)


