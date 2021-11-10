import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader
from Functions.Event.Event import EventGenerator, Event
from Functions.Tools.Alerting import Notification
import torch
from skhep.math.vectors import LorentzVector

class CreateEventGraph:

    def __init__(self, Event):
        self.G = nx.Graph()
        self.Truth_G = nx.Graph()
        self.Event = Event
        self.EdgeAttributes = []
        self.NodeAttributes = []
        self.ExcludeSelfLoops = True
        self.DefaultEdgeWeight = 1

    def CreateParticleNodes(self):
        self.Nodes = []
        for i in range(len(self.Event)):
            self.Nodes.append(i)
        self.G.add_nodes_from(self.Nodes)
        
        for i in range(len(self.Event)):
            self.G.nodes[i]["ParticleIndex"] = self.Event[i].Index
            self.G.nodes[i]["ParticleType"] = self.Event[i].Type

    def CreateParticlesEdgesAll(self):
        self.Edges = []
        for i in self.Nodes:
            for j in self.Nodes:
                if self.ExcludeSelfLoops and i == j:
                    continue
                self.Edges.append((i, j))
        self.G.add_edges_from(self.Edges)

    def CreateDefaultEdgeWeights(self):
        
        for i in self.Edges:
            i_e = i[0]
            i_j = i[1]
            self.G[i_e][i_j]["weight"] = self.DefaultEdgeWeight
    
    def CalculateEdgeAttributes(self, fx):
        
        for i in range(len(self.NodeAttributes)):
            attr = self.NodeAttributes[i]
            d_label = str("d_" + attr)
            for n in self.Nodes:
                for k in self.Nodes:
                    if k == n and self.ExcludeSelfLoops:
                        continue
                    self.G[n][k][d_label] = [fx(self.Event[k], self.Event[n], attr)]
            self.EdgeAttributes.append(d_label)

    def CalculateNodeAttributes(self):
        for i in self.NodeAttributes:
            if i == "":
                continue
            l = {}
            for n in self.Nodes:
                l[n] = [torch.tensor(getattr(self.Event[n], i), dtype = torch.float)]
            nx.set_node_attributes(self.G, l, name = i)
    
    def CalculateNodeDifference(self):
        def fx(a, b, attr):
            a = getattr(a, attr)
            b = getattr(b, attr)
            return a-b
        self.CalculateEdgeAttributes(fx)
        self.CalculateNodeAttributes()
    
    def CalculateNodeMultiplication(self):
        def fx(a, b, attr):
            a = getattr(a, attr)
            b = getattr(b, attr)
            return a*b        
        self.CalculateEdgeAttributes(fx)
        self.CalculateNodeAttributes()
    
    def CalculateParticledR(self):
        def fx(a, b, attr):
            return a.DeltaR(b)
        self.CalculateEdgeAttributes(fx)
        self.CalculateNodeAttributes()

    def CalculateNodeMultiplicationIndex(self):
        def fx(a, b, attr):
            res = a.Index
            if a.Index != b.Index:
                res = -1
            return res
        self.CalculateEdgeAttributes(fx)
        self.CalculateNodeAttributes()

    def CalculateInvariantMass(self):
        def fx(a, b, attr):
            vec1 = LorentzVector()
            vec2 = LorentzVector()
            vec1.setptetaphie(a.pt, a.eta, a.phi, a.e)
            vec2.setptetaphie(b.pt, b.eta, b.phi, b.e)
            ab = vec1+vec2
            return ab.mass
        self.CalculateEdgeAttributes(fx)

    def CalculationProxy(self, Dict):
        for i in Dict:
            self.NodeAttributes = [i]
            if Dict[i] == "Diff":
                self.CalculateNodeDifference()
            
            if Dict[i] == "Multi":
                self.CalculateNodeMultiplication()
            
            if Dict[i] == "MultiIndex":
                self.CalculateNodeMultiplicationIndex()

            if Dict[i] == "dR":
                self.CalculateParticledR()
            
            if Dict[i] == "":
                self.CalculateNodeAttributes()

            if Dict[i] == "invMass":
                self.CalculateInvariantMass()

    def ConvertToData(self):
        # Need to figure out why I cant edit input attributes as described here: 
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
        return from_networkx(self.G) 

class GenerateDataLoader(Notification):

    def __init__(self, Bundle):
        if isinstance(Bundle, EventGenerator):
            self.__Events = Bundle.Events
        self.ExcludeAnomalies = False
        
        self.Verbose = True
        Notification.__init__(self, self.Verbose)
        self.Caller = "GenerateDataLoader"
        
        #if torch.cuda.is_available():
        #    self.Device = torch.device("cuda")
        #else:
        #    self.Device = torch.device("cpu")

        self.Device = torch.device("cpu")
        self.NodeAttributes = {}
        self.TruthAttribute = {}

        self.DataLoader = None
        self.DefaultBatchSize = None
        self.ParticleLevel = "TruthTop"

    def GetEventParticles(self, Ev, Branch):
        out = {}
        if hasattr(Ev, Branch):
            out = getattr(Ev, Branch)
        if isinstance(out, dict):
            return False
        return out

    def CreateEventData(self, Event, EventIndex, name):
        ED = CreateEventGraph(Event)
        ED.CreateParticleNodes()
        ED.CreateParticlesEdgesAll() 
        ED.CalculationProxy(self.NodeAttributes)
        ED = ED.ConvertToData()
        
        Node_Merge = []
        Edge_Merge = []
        for i in self.NodeAttributes:
            if hasattr(ED, i):
                Node_Merge.append(getattr(ED,i).T[0])
            if hasattr(ED, "d_"+i):
                Edge_Merge.append(getattr(ED, "d_"+i).T[0])
        
        if len(Node_Merge) != 0:
            ten = torch.stack(Node_Merge)
            setattr(ED, "x", ten.T)
        if len(Edge_Merge) != 0:
            ten = torch.stack(Edge_Merge)
            setattr(ED, "edge_attr", ten)
        setattr(ED, "EventIndex", EventIndex)

        if name != None:
            setattr(ED, "Name", name)

        return ED
    
    def AssignTruthLabel(self, Event):
        ET = CreateEventGraph(Event)
        ET.CreateParticleNodes()
        ET.CreateParticlesEdgesAll()
        ET.CalculationProxy(self.TruthAttribute)
        D = ET.ConvertToData()
        for i in self.TruthAttribute:
            at = getattr(D, i)
            return at.T[0].long()

    def TorchDataLoader(self, branch = "nominal", name = None):
        self.EventData = {}
        self.TruthLoader = []
            
        self.Notify("CONVERTING")
        for i in self.__Events:
            e = self.__Events[i][branch]

            if e.BrokenEvent:
                continue
            
            if self.ExcludeAnomalies == "TruthMatch" and self.Anomaly_TruthMatch:
                continue
            if self.ExcludeAnomalies == "Detector" and self.Anomaly_Detector:
                continue
            
            obj = []
            if self.ParticleLevel == "TruthTop":
                obj += e.TruthTops

            if self.ParticleLevel == "TruthTopChildren":
                obj += e.TruthChildren
            
            if self.ParticleLevel == "TruthJets":
                obj += e.TruthJets
                obj += e.Electrons
                obj += e.Muons

            if self.ParticleLevel == "Detector":
                obj += e.Electrons
                obj += e.Muons
                obj += e.Jets

            if self.ParticleLevel == "DetectorRCJets":
                obj += e.RCJets
                obj += e.Muons
                obj += e.Electrons
                
            data = self.CreateEventData(obj, i, name)
            data.y = self.AssignTruthLabel(obj)
            data.mask = torch.ones(data.y.shape, dtype = torch.bool)
            data.to(self.Device)

            obj_len = len(obj)

            if obj_len in self.EventData:
                self.EventData[obj_len].append(data)
            else:
                self.EventData[obj_len] = []
                self.EventData[obj_len].append(data)

        self.DataLoader = {}
        for i in self.EventData:
            if self.DefaultBatchSize == None:
                batch_s = len(self.EventData[i])
            else:
                batch_s = self.DefaultBatchSize
            self.DataLoader[i] = DataLoader(self.EventData[i], batch_size = batch_s)
        
        self.Notify("COMPLETE")   
