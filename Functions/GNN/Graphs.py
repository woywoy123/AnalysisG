from Functions.Event.EventGenerator import EventGenerator
from Functions.Tools.Alerting import Notification

from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import torch
import networkx as nx
import numpy as np
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import RandomSampler

class EventGraph:

    def __init__(self, Event, Level, Tree):
        self.G = nx.Graph()
        self.Particles = []
        self.SelfLoop = False
        self.NodeParticleMap = {}
        self.Nodes = []
        self.Edges = []
        self.EdgeAttr = {}
        self.NodeAttr = {}
        self.Event = Event[Tree]
        self.iter = -1
        
        if Level == "JetLepton":
            self.Particles += self.Event.Jets
            self.Particles += self.Event.Muons
            self.Particles += self.Event.Electrons

        if Level == "RCJetLepton":
            self.Particles += self.Event.RCJets
            self.Particles += self.Event.Muons
            self.Particles += self.Event.Electrons

        if Level == "TruthTops":
            self.Particles += self.Event.TruthTops

        if Level == "TruthChildren_init":
            self.Particles += self.Event.TruthChildren_init

        if Level == "TruthChildren":
            self.Particles += self.Event.TruthChildren

        if Level == "TopPostFSR":
            self.Particles += self.Event.TopPostFSR

        if Level == "TopPostFSRChildren":
            self.Particles += self.Event.TopPostFSRChildren

        if Level == "TopPreFSR":
            self.Particles += self.Event.TopPreFSR

        if Level == "TruthJetsLep":
            self.Particles += self.Event.TruthJets
            self.Particles += self.Event.Electrons
            self.Particles += self.Event.Muons

        if Level == "TruthTopChildren":
            self.Particles += self.Event.TruthTopChildren
        
    def CleanUp(self):
        self.NodeAttr = {}
        self.EdgeAttr = {}
        del self.Event
        del self.Nodes 
        del self.Edges
        del self.Particles
        del self.NodeParticleMap
        del self.G

    def CreateParticleNodes(self):
        for i in range(len(self.Particles)):
            self.Nodes.append(i)
            self.NodeParticleMap[i] = self.Particles[i]
        self.G.add_nodes_from(self.Nodes)

    def CreateEdges(self):
        for i in self.Nodes:
            for j in self.Nodes:
                if self.SelfLoop == False and i == j:
                    continue 
                self.Edges.append([i, j])
        self.G.add_edges_from(self.Edges)

    def ConvertToData(self):
        edge_index = torch.tensor(self.Edges, dtype=torch.long).t().contiguous()
        self.Data = Data(edge_index = edge_index)

        # Apply Node Features [NODES X FEAT]
        for i in self.NodeAttr:
            fx = self.NodeAttr[i]
            attr_v = []
            for n in self.NodeParticleMap:
                p = self.NodeParticleMap[n]
                attr_i = []
                for fx_i in fx:
                    attr_i.append(fx_i(p))
                attr_v.append(attr_i)
            attr_ten = torch.tensor(attr_v)
            setattr(self.Data, i, attr_ten)
        
    
        # Apply Edge Features [EDGES X FEAT]
        for i in self.EdgeAttr:
            fx = self.EdgeAttr[i]
            attr_v = []
            for n in self.Edges:
                p_i = self.NodeParticleMap[n[0]]
                p_j = self.NodeParticleMap[n[1]]
                attr_i = []
                for fx_i in fx:
                    attr_i.append(fx_i(p_i, p_j))
                attr_v.append(attr_i)
            attr_ten = torch.tensor(attr_v, dtype = torch.float)
            setattr(self.Data, i, attr_ten)
        setattr(self.Data, "i", self.iter)  
        self.Data.num_nodes = len(self.Particles)

    def SetNodeAttribute(self, c_name, fx):
        if c_name not in self.NodeAttr:
            self.NodeAttr[c_name] = []
        self.NodeAttr[c_name].append(fx)

    def SetEdgeAttribute(self, c_name, fx):
        if c_name not in self.EdgeAttr:
            self.EdgeAttr[c_name] = []
        self.EdgeAttr[c_name].append(fx)



class GenerateDataLoader(Notification):
    
    def __init__(self):
        self.Verbose = True 
        self.Debug = False
        
        Notification.__init__(self, self.Verbose)
        self.Caller = "GenerateDataLoader"
        self.Device_s = "cuda"

        self.Bundles = []
        self.DataLoader = {}
        self.EventMap = {}
        self.TestDataLoader = {}
        self.__iter = 0

        self.TestSize = 40
        self.ValidationTrainingSize = 60

        self.SelfLoop = False
        self.Converted = False
        self.TrainingTestSplit = False
        self.Processed = False
        self.Trained = False

        self.EdgeAttribute = {}
        self.NodeAttribute = {}
        self.EdgeTruthAttribute = {}
        self.NodeTruthAttribute = {}

    def AddEdgeFeature(self, name, fx):
        self.EdgeAttribute[name] = fx

    def AddNodeFeature(self, name, fx):
        self.NodeAttribute[name] = fx 

    def AddNodeTruth(self, name, fx):
        self.NodeTruthAttribute[name] = fx

    def AddEdgeTruth(self, name, fx):
        self.EdgeTruthAttribute[name] = fx

    def AddSample(self, Bundle, Tree, Level = "JetLepton"):
        self.Device = torch.device(self.Device_s)
        if isinstance(Bundle, EventGenerator) == False:
            self.Warning("SKIPPED :: NOT A EVENTGENERATOR OBJECT!!!")
            return False
        for i in Bundle.FileEventIndex: 
            self.Notify("ADDING SAMPLE -> (" + Tree + ") " + i)

        self.Notify("DATA WILL BE PROCESSED ON: " + self.Device_s)
        start = self.__iter
        self.len = len(Bundle.Events)
        for it in Bundle.Events:
            
            e = EventGraph(Bundle.Events[it], Level, Tree)
            e.SelfLoop = self.SelfLoop
            e.iter = self.__iter
            e.CreateParticleNodes()
            n_particle = len(e.Particles)

            if n_particle <= 1:
                self.Warning("EMPTY EVENT")
                continue
            
            e.CreateEdges()
            Bundle.Events[it] = -1

            for i in self.NodeAttribute:
                e.SetNodeAttribute(i, self.NodeAttribute[i])
            
            for i in self.EdgeAttribute:
                e.SetEdgeAttribute(i, self.EdgeAttribute[i])

            for i in self.NodeTruthAttribute:
                e.SetNodeAttribute("y", self.NodeTruthAttribute[i])

            for i in self.EdgeTruthAttribute:
                e.SetEdgeAttribute("edge_y", self.EdgeTruthAttribute[i])
            e.ConvertToData()
            
            if self.Debug:
                pass
            else:
                e.CleanUp()
                e.Data.to(self.Device_s, non_blocking=True)
                e = e.Data

            self.ProgressInformation("CONVERSION")
            
            if n_particle not in self.DataLoader:
                self.DataLoader[n_particle] = []
            self.DataLoader[n_particle].append(e)
            self.EventMap[self.__iter] = e 
            self.__iter += 1
        
        self.ResetAll()
        self.Bundles.append([Tree, Bundle, start, self.__iter-1, Level])
        self.Converted = True
        self.Notify("FINISHED CONVERSION")

    def MakeTrainingSample(self):
        self.Notify("WILL SPLIT DATA INTO TRAINING/VALIDATION (" + str(self.ValidationTrainingSize) + "%) " + "- TEST (" + str(self.TestSize) + "%) SAMPLES")
        
        All = np.array(list(self.EventMap))
        rs = ShuffleSplit(n_splits = 1, test_size = float(self.TestSize/100), random_state = 42)
        for train_idx, test_idx in rs.split(All):
            pass
        
        for i in train_idx:
            i = self.EventMap[i]
            n_particle = i.num_nodes
            if n_particle not in self.DataLoader:
                self.DataLoader[n_particle] = []
            self.DataLoader[n_particle].append(i)
        
        self.TestDataLoader = {}
        for i in test_idx:
            i = self.EventMap[i]
            n_particle = i.num_nodes
            if n_particle not in self.TestDataLoader:
                self.TestDataLoader[n_particle] = []
            self.TestDataLoader[n_particle].append(i)
        self.TrainingTestSplit = True

        del self.EdgeAttribute
        del self.NodeAttribute
        del self.NodeTruthAttribute
        del self.EdgeTruthAttribute
        del self.EventMap

