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

        if Level == "TruthChildren_init_NoLep":
            for i in self.Event.TruthChildren_init:
                if abs(i.pdgid) not in [11, 12, 13, 14, 15, 16]:
                    self.Particles.append(i)
        
        if Level == "TruthChildren_NoLep":
            for i in self.Event.TruthChildren:
                if abs(i.pdgid) not in [11, 12, 13, 14, 15, 16]:
                    self.Particles.append(i)

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

    def CleanUp(self):
        self.NodeAttr = {}
        self.EdgeAttr = {}
        del self.Event
        del self.Nodes 
        del self.Edges
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
        Notification.__init__(self, self.Verbose)
        self.Caller = "GenerateDataLoader"
        
        if torch.cuda.is_available():
            self.Device = torch.device("cuda")
            self.Device_s = "cuda"
        else:
            self.Device = torch.device("cpu")
            self.Device_s = "cpu"

        self.Bundles = []
        self.EventData = {}
        self.EventMap = {}
        self.EventTestData = {}
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
                del e
                continue

            e.CreateEdges()
            ev = Bundle.Events[it]
            Bundle.Events[it] = -1
            del ev

            for i in self.NodeAttribute:
                e.SetNodeAttribute(i, self.NodeAttribute[i])
            
            for i in self.EdgeAttribute:
                e.SetEdgeAttribute(i, self.EdgeAttribute[i])

            for i in self.NodeTruthAttribute:
                e.SetNodeAttribute("y", self.NodeTruthAttribute[i])

            for i in self.EdgeTruthAttribute:
                e.SetEdgeAttribute("edge_y", self.EdgeTruthAttribute[i])
            e.ConvertToData()
            
            self.ProgressInformation("CONVERSION")
            
            if n_particle not in self.EventData:
                self.EventData[n_particle] = []
            self.EventData[n_particle].append(e)
            self.EventMap[self.__iter] = e 
            self.__iter += 1
        
        self.ResetAll()
        self.Bundles.append([Tree, Bundle, start, self.__iter-1, Level])

    def MakeTrainingSample(self):
        self.Notify("WILL SPLIT DATA INTO TRAINING/VALIDATION (" + str(self.ValidationTrainingSize) + "%) " + "- TEST (" + str(self.TestSize) + "%) SAMPLES")
        
        All = []
        for i in self.EventData:
            All += self.EventData[i]
        All = np.array(All)
        
        rs = ShuffleSplit(n_splits = 1, test_size = float(self.TestSize/100), random_state = 42)
        for train_idx, test_idx in rs.split(All):
            pass

        self.EventData = {}
        for i in All[train_idx]:
            n_particle = len(i.Particles)
            if n_particle not in self.EventData:
                self.EventData[n_particle] = []
            self.EventData[n_particle].append(i)
        
        for i in All[test_idx]:
            n_particle = len(i.Particles)
            if n_particle not in self.EventTestData:
                self.EventTestData[n_particle] = []
            self.EventTestData[n_particle].append(i)
        
        self.TrainingTestSplit = True

    def ToDataLoader(self):
        self.Notify("CONVERTING EVENTS TO PyGeometric DATA")
        self.DataLoader = {}
        for i in self.EventData:
            Data = []
            for k in self.EventData[i]:
                Data.append(k.Data.to(self.Device_s, non_blocking=True))
                k.CleanUp()
            self.DataLoader[i] = Data
        
        self.TestDataLoader = {}
        for i in self.EventTestData:
            Data = []
            for k in self.EventTestData[i]:
                Data.append(k.Data.to(self.Device_s, non_blocking=True))
                k.CleanUp()
            self.TestDataLoader[i] = Data
        
        self.Converted = True
        self.Notify("FINISHED CONVERSION")

        del self.EdgeAttribute
        del self.NodeAttribute
        del self.NodeTruthAttribute
        del self.EdgeTruthAttribute

