from Functions.Tools.Alerting import Notification
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Models import EdgeConv, InvMassGNN, PathNet

import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import accuracy

import numpy as np
from sklearn.model_selection import KFold 

import copy
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

        self.LearningRate = 1e-2
        self.WeightDecay = 1e-4
        self.DefaultBatchSize = 1
        self.kFold = 10
        self.Model = None

        self.LossTrainStatistics = {}
        self.TrainStatistics = {}
        
        self.ValidationStatistics = {}
        self.LossValidationStatistics = {}

        self.OptimizerSnapShots = {}

        self.epoch = 0
        self.Trained = False

    def kFoldTraining(self):
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state = 42)
        for epoch in range(self.Epochs):
            self.Notify("EPOCH =============================== " +str(epoch+1) + "/" + str(self.Epochs))
            self.epoch = epoch+1

            self.LossTrainStatistics[self.epoch] = []
            self.LossValidationStatistics[self.epoch] = []

            self.ValidationStatistics[self.epoch] = []
            self.TrainStatistics[self.epoch] = []
 
            for n_node in self.DataLoader:
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
                    self.ResetAll()
                    self.len = len(train_loader)
                    for ts in train_loader:
                        self.sample = ts
                        self.TrainClassification()
                        self.ProgressInformation(Mode = "LEARNING")
                    
                    self.Notify("------- Collecting Accuracy/Loss -------")
                    TA = self.GetClassificationAccuracy(train_loader)
                    TL = self.GetClassificationLoss(train_loader)
                    
                    self.Notify("-------> Validation <-------")
                    self.Notify("------- Collecting Accuracy/Loss -------")                   
                    VA = self.GetClassificationAccuracy(valid_loader)
                    VL = self.GetClassificationLoss(valid_loader)
                    
                    self.TrainStatistics[self.epoch].append(TA)
                    self.ValidationStatistics[self.epoch].append(VA)

                    self.LossTrainStatistics[self.epoch].append(TL)
                    self.LossValidationStatistics[self.epoch].append(VL)

                self.Notify("CURRENT LOSS FUNCTION: " + str(round(float(self.L), 7)))
                self.Notify("CURRENT ACCURACY (Validation): " + str(round(float(VA), 7)))
            self.OptimizerSnapShots[self.epoch] = copy.deepcopy(self.Optimizer)
        self.Loader.Trained = True 
    
    def GetClassificationAccuracy(self, loader):
        self.ResetAll()
        self.len = len(loader)
        self.Model.eval()
        P = []
        T = []
        l = 0
        for i in loader:
            _, pred = self.Model(i).max(1)
            truth = i.y.t().contiguous().squeeze()
            
            if l == 0:
                l = len(truth.tolist()) 
            
            if len(truth.tolist()) == l and len(pred.tolist()) == l:
                T.append(truth.tolist())
                P.append(pred.tolist())
            self.ProgressInformation("ACCURACY")

        p = accuracy(torch.tensor(P), torch.tensor(T))
        return p
    
    def GetClassificationLoss(self, loader):
        self.ResetAll()
        self.len = len(loader)
        self.Model.eval()
        L = []
        for i in loader:
            pred = self.Model(i)
            Loss = torch.nn.CrossEntropyLoss()
            truth = i.y.t().contiguous().squeeze()

            L.append(float(Loss(pred, truth)))
            self.ProgressInformation("LOSS")
        return L

    def DefineEdgeConv(self, in_channels, out_channels):
        self.Classifier = True
        self.Model = EdgeConv(in_channels, out_channels)
        self.DefineOptimizer()

    def DefineInvMass(self, out_channels):
        self.Classifier = True
        self.Model = InvMassGNN(out_channels)
        self.DefineOptimizer()

    def DefinePathNet(self, PCut = 0.5, complex = 64, path = 64, hidden = 64, out = 20):
        self.Classifier = True
        self.Model = PathNet(PCut, complex, path, hidden, out)
        self.DefineOptimizer()

    def TrainClassification(self):
        self.Model.train()
        self.Optimizer.zero_grad()
        
        pred = self.Model(self.sample)
        Loss = torch.nn.CrossEntropyLoss()
        self.L = Loss(pred, self.sample.y.t().contiguous().squeeze())
        self.L.backward()
        self.Optimizer.step()

    def DefineOptimizer(self):
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
    
    def ApplyToDataSample(self, sample, attr):
        DataLoader = {}
        if isinstance(sample, dict):
            DataLoader = sample
        elif isinstance(sample, GenerateDataLoader) and sample.Converted == True:
            DataLoader = sample.DataLoader
        elif isinstance(sample, GenerateDataLoader):
            sample.ToDataLoader()
            DataLoader = sample.DataLoader
        else:
            self.Warning("FAILURE::WRONG DATALOADER OBJECT")
            return False

        for n_part in sample.EventData:
            for i in sample.EventData[n_part]:
                Data = i.Data.to(self.Device_s, non_blocking = True)
                Event_Obj = i.Event

                _, pred = self.Model(Data).max(1)

                for n in range(len(pred)):
                    setattr(i.NodeParticleMap[n], attr, pred[n])
        sample.Processed = True 
