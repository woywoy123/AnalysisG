from Functions.Tools.Alerting import Notification
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Models import EdgeConv
from torch_geometric.loader import DataLoader

import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold 


class Optimizer(Notification):

    def __init__(self, Loader):
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
        
        self.Epochs = 50
        self.Device = Loader.Device
        self.Device_s = Loader.Device_s
        
        self.LearningRate = 1e-2
        self.WeightDecay = 1e-4
        self.DefaultBatchSize = 1
        self.kFold = 10
        self.Model = None

        self.EdgeAttribute = Loader.EdgeAttribute
        self.NodeAttribute = Loader.NodeAttribute

        self.LossStatistics = {}
        self.epoch = 0

    def kFoldTraining(self):
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state = 42)
        for epoch in range(self.Epochs):
            self.Notify("EPOCH: " +str(epoch+1) + "/" + str(self.Epochs))
            self.epoch = epoch
            self.LossStatistics[epoch] = []
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
                    
                    for ts in train_loader:
                        self.sample = ts
                        self.TrainClassification()

                    for vl in valid_loader:
                        self.Model.eval()
                        pred = self.Model(vl.x, vl.edge_index)

                self.Notify("CURRENT LOSS FUNCTION: " + str(round(float(self.L), 3)))
                pred = self.Model(self.sample.x, self.sample.edge_index)
                print(torch.stack(list(pred)), "|",  self.sample.y)

    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.__DefineOptimizer()

    def TrainClassification(self):
        self.Model.train()
        self.Optimizer.zero_grad()
        Loss = torch.nn.CrossEntropyLoss()
        
        pred = self.Model(self.sample.x, self.sample.edge_index)
        self.L = Loss(pred, self.sample.y)
        self.L.backward()
        self.Optimizer.step()

        self.LossStatistics[self.epoch].append(self.L)

    def __DefineOptimizer(self):
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        
