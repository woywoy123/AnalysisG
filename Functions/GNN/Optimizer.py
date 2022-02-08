from Functions.Tools.Alerting import Notification
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Models import EdgeConv, InvMassGNN, PathNet
from Functions.IO.Files import WriteDirectory

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
        self.DefaultBatchSize = 10
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
        if TargetType == "Edges":
            edge_index = Sample.edge_index
            p = self.Model.Adj_M[edge_index[0], edge_index[1]]
            return p, torch.round(p).to(torch.int), torch.tensor(Sample.y[edge_index[0]] == Sample.y[edge_index[1]], dtype= torch.float).t()[0]

    def DefineLossFunction(self, LossFunction):
        if LossFunction == "CrossEntropyLoss":
            self.LossFunction = torch.nn.CrossEntropyLoss()
        elif LossFunction == "MSELoss":
            self.LossFunction = torch.nn.MSELoss()
        self.DefaultLossFunction = LossFunction

    def DefineOptimizer(self):
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
    
    # Need to rebuild this!!!!
    #def ApplyToDataSample(self, sample, attr):
    #    DataLoader = {}
    #    if isinstance(sample, dict):
    #        DataLoader = sample
    #    elif isinstance(sample, GenerateDataLoader) and sample.Converted == True:
    #        DataLoader = sample.DataLoader
    #    elif isinstance(sample, GenerateDataLoader):
    #        sample.ToDataLoader()
    #        DataLoader = sample.DataLoader
    #    else:
    #        self.Warning("FAILURE::WRONG DATALOADER OBJECT")
    #        return False

    #    for n_part in sample.EventData:
    #        for i in sample.EventData[n_part]:
    #            Data = i.Data.to(self.Device_s, non_blocking = True)
    #            Event_Obj = i.Event

    #            _, pred = self.Model(Data).max(1)

    #            for n in range(len(pred)):
    #                setattr(i.NodeParticleMap[n], attr, pred[n])
    #    sample.Processed = True 

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
        elif Target == "Edges": 
            self.DefineLossFunction("MSELoss")

    def DefinePathNet(self, complex = 64, path = 64, hidden = 64, out = 50, Target = "Nodes"):
        self.Classifier = True
        self.Model = PathNet(complex, path, hidden, out, self.Debug)
        self.DefineOptimizer()

        self.DefaultTargetType = Target
        if Target == "Nodes":
            self.DefineLossFunction("CrossEntropyLoss")
        elif Target == "Edges": 
            self.DefineLossFunction("MSELoss")
