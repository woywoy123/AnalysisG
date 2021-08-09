import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader
from Functions.Event.Event import EventGenerator, Event

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
                    
                    self.G[n][k][d_label] = fx(self.Event[k], self.Event[n], attr)

            self.EdgeAttributes.append(d_label)

    def CalculateNodeDifference(self):
        def fx(a, b, attr):
            a = getattr(a, attr)
            b = getattr(b, attr)
            return a-b
        self.CalculateEdgeAttributes(fx)
    
    def CalculateNodeMultiplication(self):
        def fx(a, b, attr):
            a = getattr(a, attr)
            b = getattr(b, attr)
            return a*b        
        self.CalculateEdgeAttributes(fx)

    def CalculateParticledR(self):
        def fx(a, b, attr):
            return a.DeltaR(b)
        self.CalculateEdgeAttributes(fx)

    def CalculateNodeMultiplicationIndex(self):
        def fx(a, b, attr):
            res = a.Index
            if a.Index != b.Index:
                res = -99
            return res
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

    def ConvertToData(self):
        # Need to figure out why I cant edit input attributes as described here: 
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
        return from_networkx(self.G) 

class GenerateDataLoader:

    def __init__(self, Bundle):
        if isinstance(Bundle, EventGenerator):
            self.__Events = Bundle.Events
        self.ExcludeAnomalies = False
        self.NodeAttributes = {}
        self.DataLoader = None
        self.TruthLoader = None
        self.DefaultBatchSize = 32

    def GetEventParticles(self, Ev, Branch):
        out = {}
        if hasattr(Ev, Branch):
            out = getattr(Ev, Branch)

        if isinstance(out, dict):
            return False

        return out

    def CreateEventData(self, Event):
        ED = CreateEventGraph(Event)
        ED.CreateParticleNodes()
        ED.CreateParticlesEdgesAll() 
        ED.CalculationProxy(self.NodeAttributes)
        return ED

    def CreateEventTopTruth(self, Event):
        ED = CreateEventGraph(Event)
        ED.CreateParticleNodes()
        ED.CreateParticlesEdgesAll() 
        ED.CalculationProxy({"Signal" : "Multi"})
        return ED 

    def CreateEventTruth(self, Event):
        ED = CreateEventGraph(Event)
        ED.CreateParticleNodes()
        ED.CreateParticlesEdgesAll()
        ED.CalculationProxy({"Signal": "MultiIndex"})
        for i in ED.G.edges:
            if ED.G[i[0]][i[1]]["d_Signal"] == -99:
                ED.G.remove_edge(i[0], i[1])
        return ED

    def TorchDataLoader(self, branch = "nominal"):
        self.Loader = []
        self.TruthLoader = []
        for i in self.__Events:
            e = self.__Events[i][branch]

            if e.BrokenEvent:
                continue
            
            if self.ExcludeAnomalies == "TruthMatch" and self.Anomaly_TruthMatch:
                continue
            if self.ExcludeAnomalies == "Detector" and self.Anomaly_Detector:
                continue
            
            # Means this is the truth matched detector stuff 
            if hasattr(e, "CallLoop"):
                All = []
                All += e.Muons
                All += e.Electrons
                All += e.Jets
                self.Loader.append(self.CreateEventData(All))   
                self.TruthLoader.append(self.CreateEventTruth(All)) 
 
            # Get truth Children
            elif self.GetEventParticles(e, "TruthChildren") != False:
                obj = e.TruthChildren
                self.Loader.append(self.CreateEventData(obj))   
                self.TruthLoader.append(self.CreateEventTruth(obj)) 


            # Get truth tops
            elif self.GetEventParticles(e, "TruthTops") != False:
                obj = e.TruthTops
                self.Loader.append(self.CreateEventData(obj))   
                self.TruthLoader.append(self.CreateEventTopTruth(obj)) 
 

           
            break

        if len(self.Loader) != 0:
            for i in self.Loader:
                i.ConvertToData()
            self.DataLoader = DataLoader(self.Loader, batch_size = self.DefaultBatchSize)
        if len(self.TruthLoader) != 0:
            for i in self.TruthLoader:
                i.ConvertToData()
            self.DataTruthLoader = DataLoader(self.TruthLoader, batch_size = self.DefaultBatchSize)

