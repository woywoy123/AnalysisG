from Functions.GNN.Metrics import EvaluationMetrics
from Functions.Tools.Alerting import Notification
import torch 
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from Functions.GNN.GNN_Models import EdgeConv

# Training and dataset management 
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, SubsetRandomSampler
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt



class Optimizer(Notification):

    def __init__(self, DataLoaderObject = None):
        self.Optimizer = ""
        self.Model = ""
        
        verbose = True
        if DataLoaderObject != None:
            self.DataLoader = DataLoaderObject.DataLoader
            self.Device = DataLoaderObject.Device
            self.LoaderObject = DataLoaderObject
            verbose = DataLoaderObject.Verbose
        else:
            self.DataLoader = {}
        Notification.__init__(self, verbose)

        self.Epochs = 500
        self.LearningRate = 0.001
        self.WeightDecay = 1e-4
        self.kFold = 10
        self.DefaultBatchSize = 5000
        
        self.Caller = "OPTIMIZER"

    def SampleHandler(self, List):
        if len(List) != 0:
            self.Device = List[0].Device
            self.Verbose = List[0].Verbose
        
        self.DataLoader = {}
        for i in List:
            for k in i.DataLoader:
                if k in self.DataLoader:
                    self.DataLoader[k] += i.EventData[k]
                else:
                    self.DataLoader[k] = []
                    self.DataLoader[k] += i.EventData[k]
        
        for i in self.DataLoader:
            if len(self.DataLoader[i]) < self.kFold and len(self.DataLoader[i]) > 4:
                self.kFold = len(self.DataLoader[i])
    
    def KFoldTraining(self):
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state = 42)
        
       
        for n_node in self.DataLoader:
            self.Notify("NUMBER OF NODES -----> " + str(n_node) + " BEING TESTED")
            CurrentData = self.DataLoader[n_node] 
            for epoch in range(self.Epochs):
                self.Notify("EPOCH: " +str(epoch+1) + "/" + str(self.Epochs))
                for fold, (train_idx, val_idx) in enumerate(Splits.split(np.arange(len(CurrentData)))):
                    self.Notify("CURRENT k-Fold: " + str(fold))
                    
                    train_sampler = SubsetRandomSampler(train_idx)
                    test_sampler = SubsetRandomSampler(val_idx)

                    train_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = train_sampler)
                    test_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = test_sampler)
                    
                    for tr in train_loader:
                        tr.to(self.Device)
                        self.data = tr
                        self.Learning()
                    
                    self.Notify("TRANING::CURRENT LOSS: " + str(float(self.L))) 
                    
                    val_cor = 0
                    val_sam = 0
                    for ts in test_loader:
                        p = self.Prediction(ts)
                        val_cor += (p == ts.y).sum().item()
                        val_sam += len(test_loader.sampler)
                    
                    self.Notify("VALIDATION::CORRECT: -> " + str(float(val_cor/val_sam)*100))


    def EpochLoop(self):
        self.Notify("EPOCHLOOP::Training")
        
        lossArr = []
        
        for epoch in range(self.Epochs):
            for n in self.DataLoader:
                self.DataLoader[n].shuffle = False
                for data in self.DataLoader[n]:
                    data.to(self.Device)
                    self.data = data
                    self.Learning()
            self.Notify("EPOCHLOOP::Training::EPOCH " + str(epoch+1) + "/" + str(self.Epochs) + " -> Current Loss: " + str(float(self.L)))
            lossArr.append(float(self.L))
         
        plt.hist(lossArr, bins=200)    


    def Learning(self):

        self.Model.train()
        self.Optimizer.zero_grad()
        Loss = torch.nn.CrossEntropyLoss()
        
        x = self.Model(self.data.x, self.data.edge_index) 
        self.L = Loss(x[self.data.mask], self.data.y[self.data.mask])
        self.L.backward()
        self.Optimizer.step()

    def Prediction(self, in_channel):
        self.Model.eval()
        Output = self.Model(in_channel.x, in_channel.edge_index)
        _, y_p = Output.max(1)
        return y_p

    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
   
    def AssignPredictionToEvents(self, EventGeneratorContainer, SystematicTree, Sample = None):
        def AppendToList(EventParticles, out):
            for k in EventParticles:
                if isinstance(k, str):
                    continue
                else:
                    out.append(k)
            return out
        
        if Sample == None:
            Sample = self.LoaderObject.EventData

        for i in Sample:
            for l in Sample[i]:
                eN = l["EventIndex"]
                Event = EventGeneratorContainer.Events[eN][SystematicTree]
                SummedParticles = []
                SummedParticles = AppendToList(Event.TruthTops, SummedParticles)
                SummedParticles = AppendToList(Event.TruthChildren, SummedParticles)
                SummedParticles = AppendToList(Event.TruthChildren_init, SummedParticles)
                SummedParticles = AppendToList(Event.TruthJets, SummedParticles)
                SummedParticles = AppendToList(Event.Muons, SummedParticles)
                SummedParticles = AppendToList(Event.Electrons, SummedParticles)
                SummedParticles = AppendToList(Event.Jets, SummedParticles)
                SummedParticles = AppendToList(Event.RCJets, SummedParticles)

                pred = self.Prediction(l)
                
                for p, p_i, po in zip(l["ParticleType"], l["ParticleIndex"], pred):
                    for pe in SummedParticles:
                        if pe.Type == p and pe.Index == p_i:
                            setattr(pe, "ModelPredicts", po.item())




