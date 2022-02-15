from Functions.Tools.Alerting import Notification
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Models import EdgeConv, InvMassGNN, PathNet
from Functions.IO.Files import WriteDirectory, Directories
from Functions.Particles.Particles import Particle

import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import accuracy

import numpy as np
from sklearn.model_selection import KFold 

import time 


class Optimizer(Notification):

    def __init__(self, Loader, Debug = False):
        self.Verbose = True
        Notification.__init__(self, self.Verbose)
        self.Caller = "OPTIMIZER"

        self.DataLoader = {}
        if isinstance(Loader, dict):
            self.DataLoader = Loader
        elif isinstance(Loader, GenerateDataLoader) and Loader.Converted == True:
            self.DataLoader = Loader.DataLoader
        elif isinstance(Loader, GenerateDataLoader):
            self.DataLoader = Loader
            self.DataLoader.ToDataLoader()
            self.DataLoader = self.Loader.DataLoader
        else:
            self.Warning("FAILED TO INITIALIZE OPTIMIZER. WRONG DATALOADER")
            return False

        self.Loader = Loader 
        self.Epochs = 50
        if Debug == False:
            self.Device = Loader.Device
            self.Device_s = Loader.Device_s
        else:
            self.Device = torch.device("cpu")
            self.Device_s = "cpu"
        self.Debug = Debug
        self.LearningRate = 1e-2
        self.WeightDecay = 1e-4
        self.DefaultBatchSize = 25
        self.MinimumEvents = 250
        self.kFold = 10
        self.NotifyTime = 5
        self.DefaultTargetType = "Nodes"
        self.DefaultLossFunction = ""
        self.TrainingName = "UNTITLED"
        self.Model = None

        self.LossTrainStatistics = {}
        self.TrainStatistics = {}
        
        self.ValidationStatistics = {}
        self.LossValidationStatistics = {}

        self.BatchRate = []
        self.BatchTime = []

        self.epoch = 0
        self.Trained = False

    def kFoldTraining(self):
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state = 42)

        MaxNode = []
        for i in self.DataLoader:
            if self.MinimumEvents <= len(self.DataLoader[i]):
                MaxNode.append(int(i))
        MaxNode.sort(reverse = True)

        WriteDirectory().MakeDir("Models/" + self.TrainingName)
        
        TimeStart = time.time()
        for epoch in range(self.Epochs):
            self.Notify("EPOCH =============================== " +str(epoch+1) + "/" + str(self.Epochs))
            self.epoch = epoch+1

            self.LossTrainStatistics[self.epoch] = []
            self.LossValidationStatistics[self.epoch] = []

            self.ValidationStatistics[self.epoch] = []
            self.TrainStatistics[self.epoch] = []
            TimeStartEpoch = time.time()

            for n_node in MaxNode:
                if n_node == 0:
                    continue
                self.Notify("NUMBER OF NODES -----> " + str(n_node) + " BEING TESTED")
                CurrentData = self.DataLoader[n_node] 
                
                if len(CurrentData) < self.kFold:
                    self.Warning("NOT ENOUGH SAMPLES FOR EVENTS WITH " + str(n_node) + " PARTICLES :: SKIPPING")
                    continue

                for fold, (train_idx, val_idx) in enumerate(Splits.split(np.arange(len(CurrentData)))):
                    self.Notify("CURRENT k-Fold: " + str(fold+1))
                    
                    train_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = SubsetRandomSampler(train_idx))
                    valid_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = SubsetRandomSampler(val_idx)) 
                    self.Notify("-------> Training <-------")
                    self.SampleLoop("Training", train_loader)                    

                    self.Notify("-------> Validation <-------")
                    self.SampleLoop("Validation", valid_loader)
                
                self.Notify("CURRENT TRAIN LOSS FUNCTION: " + str(round(float(self.LossTrainStatistics[self.epoch][-1][-1]), 7)))
                self.Notify("CURRENT ACCURACY (Validation): " + str(round(float(self.ValidationStatistics[self.epoch][-1]), 7)))
            
            torch.save(self.Model, "Models/" + self.TrainingName + "/Model_epoch" + str(epoch +1) +".pt")
            TimeEndEpoch = time.time()
            self.BatchTime.append(TimeEndEpoch - TimeStartEpoch)
        
        TimeEnd = time.time()
        self.TrainingTime = TimeEnd - TimeStart
        self.Loader.Trained = True 

    def SampleLoop(self, Mode, sample_loader):
        Mode_b = True 
        if Mode != "Training":
            Mode_b = False

        self.ResetAll()
        TT, TP, L = [], [], []
        self.len = len(sample_loader)*self.DefaultBatchSize
        for ts in sample_loader:
            self.sample = ts
            self.TrainClassification(Mode_b)
            
            for i in range(self.DefaultBatchSize):
                self.ProgressInformation(Mode)

            TT.append(self.__Truth)
            TP.append(self.__Pred) 
            L.append(float(self.L))

            if self.Rate != -1:
                self.BatchRate.append(self.Rate)

        ac = accuracy(torch.cat(TP, dim = 0), torch.cat(TT, dim = 0))
        if Mode_b:
            self.TrainStatistics[self.epoch].append(ac)
            self.LossTrainStatistics[self.epoch].append(L)
        else:
            self.ValidationStatistics[self.epoch].append(ac)
            self.LossValidationStatistics[self.epoch].append(L)

    def TrainClassification(self, Train = True):
        if Train:
            self.Model.train()
            self.Optimizer.zero_grad()
        else:
            self.Model.eval()
        
        param, P, truth = self.TargetDefinition(self.sample, self.DefaultTargetType)
        self.__Pred = P
        self.__Truth = truth
        self.L = self.LossFunction(param, truth)

        if Train:
            self.L.backward()
            self.Optimizer.step()

    def TargetDefinition(self, Sample, TargetType = "Nodes"):
        pred = self.Model(Sample)
        if TargetType == "Nodes":
            _, p = pred.max(1)
            return pred, p, Sample.y.t().contiguous().squeeze()

        if TargetType == "NodeEdges":
            edge_index = Sample.edge_index
            p = self.Model.Adj_M[edge_index[0], edge_index[1]]
            return p, torch.round(p).to(torch.int), torch.tensor(Sample.y[edge_index[0]] == Sample.y[edge_index[1]], dtype= torch.float).t()[0]

        if TargetType == "Edges":
            edge_index = Sample.edge_index
            p = self.Model.Adj_M[edge_index[0], edge_index[1]]
            return p, torch.round(p).to(torch.int), Sample.edge_y.t()[0].to(torch.float)

    def DefineLossFunction(self, LossFunction):
        if LossFunction == "CrossEntropyLoss":
            self.LossFunction = torch.nn.CrossEntropyLoss()
        elif LossFunction == "MSELoss":
            self.LossFunction = torch.nn.MSELoss()
        self.DefaultLossFunction = LossFunction

    def DefineOptimizer(self):
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
    
    def DefineEdgeConv(self, in_channels, out_channels):
        self.Classifier = True
        self.Model = EdgeConv(in_channels, out_channels)
        self.DefineOptimizer()
        self.DefineLossFunction("CrossEntropyLoss")

    def DefineInvMass(self, out_channels, Target = "Nodes"):
        self.Classifier = True
        self.Model = InvMassGNN(out_channels)
        self.DefineOptimizer()

        self.DefaultTargetType = Target
        if Target == "Nodes":
            self.DefineLossFunction("CrossEntropyLoss")
        elif Target == "NodeEdges": 
            self.DefineLossFunction("MSELoss")
        elif Target == "Edges": 
            self.DefineLossFunction("MSELoss")

    def DefinePathNet(self, complex = 64, path = 64, hidden = 64, out = 50, Target = "Nodes"):
        self.Classifier = True
        self.Model = PathNet(complex, path, hidden, out, self.Debug)
        self.DefineOptimizer()

        self.DefaultTargetType = Target
        if Target == "Nodes":
            self.DefineLossFunction("CrossEntropyLoss")
        elif Target == "NodeEdges": 
            self.DefineLossFunction("MSELoss")
        elif Target == "Edges": 
            self.DefineLossFunction("MSELoss")

    def LoadModelState(self, Path = False):
        
        if Path == False:
            Path = "Models/" + self.TrainingName

            D = Directories("")
            Files = D.ListFilesInDir(Path) 
            high = []
            for i in Files:
                high.append(int(i.split("epoch")[1].replace(".pt", "")))
            Path = "Models/" + self.TrainingName + "/Model_epoch" + str(max(high)) +".pt"
        self.Model = torch.load(Path)


    def __RebuildClustersFromEdge(self, topo):
        edge_index = self.Sample.edge_index
        TMP = {}
        for t, e_i, e_j in zip(topo, edge_index[0], edge_index[1]):
            if t > 0:
                if int(e_i) not in TMP:
                    TMP[int(e_i)] = []
                TMP[int(e_i)].append(int(e_j))      
        TMP_L = []
        for i in TMP:
            l = TMP[i]
            l.append(i)
            TMP_L.append(list(set(l)))

        TMP_L = [list(x) for x in set(tuple(x) for x in TMP_L)]

        Output = []
        for k in TMP_L:
            Par = Particle(True)
            for j in k:
                P = Particle(True)
                P.pt = self.Sample.pt[j]
                P.e = self.Sample.e[j]
                P.phi = self.Sample.phi[j]
                P.eta = self.Sample.eta[j]
                Par.Decay.append(P)    
            Par.CalculateMassFromChildren()
            Output.append(Par)
        return Output
    
    def __RebuildParticlesFromData(self, Prediction, DataLoader):
        Output = {}
        self.ResetAll()
        l = 0
        for i in DataLoader:
            l += len(DataLoader[i]) 
        self.len = l

        for nodes in DataLoader:
            for event in DataLoader[nodes]:
                self.Sample = event
                if Prediction: 
                    _, topo, truth = self.TargetDefinition(event, self.DefaultTargetType)
                else:
                    topo = event.edge_y
                self.ProgressInformation("BUILDING/PREDICTING")
                Output[event.i] = self.__RebuildClustersFromEdge(topo) 
        if len(Output) == 0:
            self.Warning("NO DATA!")
        return Output

    def RebuildTruthParticles(self):
        self.Notify("REBUILDING TRUTH PARTICLES FROM TRAINING")
        return self.__RebuildParticlesFromData(False, self.Loader.DataLoader)

    def RebuildPredictionParticles(self):
        self.Notify("REBUILDING PREDICTION PARTICLES FROM TRAINING")
        return self.__RebuildParticlesFromData(True, self.Loader.DataLoader) 

    def RebuildTruthParticlesTestData(self):
        self.Notify("REBUILDING TRUTH PARTICLES FROM TEST DATA")
        return self.__RebuildParticlesFromData(False, self.Loader.TestDataLoader)

    def RebuildPredictionParticlesTestData(self):
        self.Notify("REBUILDING PREDICTION PARTICLES FROM TEST DATA")
        return self.__RebuildParticlesFromData(True, self.Loader.TestDataLoader) 

    def RebuildEventFileMapping(self):
        it = 0 
        File = ""
        Mapping = {}
        for i in self.Loader.Bundles:
            start = i[2]
            end = i[3]
            Events = i[1].Events
            for k in Events:
                File = i[1].EventIndexFileLookup(k)
                Mapping[it] = File
                it += 1
        return Mapping
