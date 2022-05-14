import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import accuracy

from sklearn.model_selection import KFold
import numpy as np

import time
from Functions.Tools.Alerting import Notification
from Functions.IO.Files import WriteDirectory, Directories
from Functions.IO.IO import PickleObject

class Optimizer(Notification):

    def __init__(self, DataLoaderInstance = None):
        self.Verbose = True
        Notification.__init__(self, self.Verbose)

        self.Caller = "OPTIMIZER"
        ### DataLoader Inheritence 
        self.DataLoader = DataLoaderInstance
        if self.DataLoader != None:
            self.TrainingSample = DataLoaderInstance.TrainingSample

            self.EdgeFeatures = DataLoaderInstance.EdgeAttribute
            self.NodeFeatures = DataLoaderInstance.NodeAttribute
            self.GraphFeatures = DataLoaderInstance.GraphAttribute

            self.Device_S = DataLoaderInstance.Device_S
            self.Device = DataLoaderInstance.Device
        

        ### User defined ML parameters
        self.LearningRate = 0.001
        self.WeightDecay = 0.001
        self.kFold = 10
        self.Epochs = 10
        self.BatchSize = 10
        self.Model = None
        self.RunName = "UNTITLED"
        self.RunDir = "_Models"

        ### Internal Stuff 
        self.Training = True
        self.Sample = None
        self.T_Features = {}
       
    def DumpStatistics(self):
        WriteDirectory().MakeDir(self.RunDir + "/" + self.RunName + "/Statistics")
        if self.epoch == "Done":
            PickleObject(self.Stats, "Stats_" + self.epoch, self.RunDir + "/" + self.RunName + "/Statistics")
        else:
            PickleObject(self.Stats, "Stats_" + str(self.epoch+1), self.RunDir + "/" + self.RunName + "/Statistics")
        self.MakeStats()

    def MakeStats(self):

        ### Output Information
        self.Stats = {}
        self.Stats["EpochTime"] = []
        self.Stats["BatchRate"] = []
        self.Stats["kFold"] = []
        self.Stats["FoldTime"] = []
        self.Stats["Nodes"] = []

        self.Stats["Training_Accuracy"] = {}
        self.Stats["Validation_Accuracy"] = {}
        self.Stats["Training_Loss"] = {}
        self.Stats["Validation_Loss"] = {}

        for i in self.T_Features:
            self.Stats["Training_Accuracy"][i] = []
            self.Stats["Validation_Accuracy"][i] = []
            self.Stats["Training_Loss"][i] = []
            self.Stats["Validation_Loss"][i] = []

    def __GetTruthFlags(self, inp, FEAT):
        for i in inp:
            if i.startswith("T_"):
                self.T_Features[i[2:]] = [FEAT + "_" +i, FEAT + "_" +i[2:]]

    def DefineOptimizer(self):
        self.Model.to(self.Device)
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)

    def DefineLossFunction(self, LossFunction):
        if LossFunction == "CEL":
            self.LF = torch.nn.CrossEntropyLoss()
        elif LossFunction == "MSEL":
            self.LF = torch.nn.MSELoss()
   
    def MakePrediction(self, sample, TargetAttribute, Classification):
        
        self.Model(sample)
        pred = getattr(self.Model, TargetAttribute)
        if Classification:
            _, p = pred.max(1)
        else:
            p = pred
        return pred, p

    def Train(self, sample):
        for key in self.T_Features:
            key_t = self.T_Features[key][0]
            key_f = self.T_Features[key][1]
            key_l = "L_" + key
            key_c = "C_" + key
            self.DefineLossFunction(getattr(self.Model, key_l))
            truth = getattr(sample, key_t).type(torch.LongTensor).to(self.Device)
            
            if self.Training:
                self.Model.train()
                self.Optimizer.zero_grad()
            else:
                self.Model.eval()
          
            pred, p = self.MakePrediction(sample, key_f, key_c) 
            self.L = self.LF(pred, truth)
            acc = accuracy(p, truth)
            
            if self.Training:
                self.Stats["Training_Accuracy"][key][-1].append(acc)
                self.Stats["Training_Loss"][key][-1].append(self.L)
            else:
                self.Stats["Validation_Accuracy"][key][-1].append(acc)
                self.Stats["Validation_Loss"][key][-1].append(self.L)
            
            if self.Training:
                self.L.backward()
                self.Optimizer.step()



    def SampleLoop(self, samples):
        self.ResetAll() 
        self.len = len(samples.dataset)
        R = []
        for i in samples:
            if self.Training:
                self.ProgressInformation("TRAINING")
            else:
                self.ProgressInformation("VALIDATING")
            self.Train(i) 
            R.append(self.Rate)
        if self.AllReset:
            self.Stats["BatchRate"].append(R)

    def KFoldTraining(self):
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state= 42)
        N_Nodes = list(self.TrainingSample)
        N_Nodes.sort(reverse = True)
        self.__GetTruthFlags(self.EdgeFeatures, "E")
        self.__GetTruthFlags(self.NodeFeatures, "N")
        self.__GetTruthFlags(self.GraphFeatures, "G")
        self.MakeStats()
        self.DefineOptimizer()

        TimeStart = time.time()
        for self.epoch in range(self.Epochs):
            self.Notify("EPOCH =============================== " +str(self.epoch+1) + "/" + str(self.Epochs))
            
            TimeStartEpoch = time.time()
            for n_node in N_Nodes:
                Curr = self.TrainingSample[n_node]
                Curr_l = len(Curr)
                self.Notify("NUMBER OF NODES -----> " + str(n_node) + " NUMBER OF ENTRIES: " + str(Curr_l))

                if Curr_l < self.kFold:
                    self.Warning("NOT ENOUGH SAMPLES FOR EVENTS WITH " + str(n_node) + " PARTICLES :: SKIPPING")
                    continue

                self.Stats["FoldTime"].append([])
                self.Stats["kFold"].append([])
                for fold, (train_idx, val_idx) in enumerate(Splits.split(np.arange(Curr_l))):

                    for f in self.T_Features:
                        self.Stats["Training_Accuracy"][f].append([])
                        self.Stats["Validation_Accuracy"][f].append([])
                        self.Stats["Training_Loss"][f].append([])
                        self.Stats["Validation_Loss"][f].append([])

                    TimeStartFold = time.time()
                    self.Notify("CURRENT k-Fold: " + str(fold+1))

                    train_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(train_idx))
                    valid_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(val_idx)) 
                    self.Notify("-------> Training <-------")
                    self.Training = True
                    self.SampleLoop(train_loader)

                    self.Notify("-------> Validation <-------")
                    self.Training = False
                    self.SampleLoop(valid_loader)
                    
                    self.Stats["FoldTime"][-1].append(time.time() - TimeStartFold)
                    self.Stats["kFold"][-1].append(fold+1)
               
                self.Stats["Nodes"].append(n_node)

            self.Stats["EpochTime"].append(time.time() - TimeStartEpoch)
            self.DumpStatistics()

        self.Stats["TrainingTime"] = time.time() - TimeStart
        self.Stats.update(self.DataLoader.FileTraces)
        
        ix = 0
        self.Stats["n_Node_Files"] = [[]]
        self.Stats["n_Node_Count"] = [[]]
        for i in self.TrainingSample:
            for j in self.TrainingSample[i]:
                indx, n_nodes = j.i, j.num_nodes
                start, end = self.Stats["Start"][ix], self.Stats["End"][ix]
                if start <= indx and indx <= end:
                    pass
                else:
                    self.Stats["n_Node_Files"].append([])
                    self.Stats["n_Node_Count"].append([])
                    ix += 1

                if n_nodes not in self.Stats["n_Node_Files"][ix]:
                    self.Stats["n_Node_Files"][ix].append(n_nodes)
                    self.Stats["n_Node_Count"][ix].append(0)
                
                n_i = self.Stats["n_Node_Files"][ix].index(n_nodes)
                self.Stats["n_Node_Count"][ix][n_i] += 1
        self.Stats["BatchSize"] = self.BatchSize
        self.Stats["Model"] = {}
        self.Stats["Model"]["LearningRate"] = self.LearningRate
        self.Stats["Model"]["WeightDecay"] = self.WeightDecay
        self.Stats["Model"]["ModelFunctionName"] = type(self.Model)
        self.epoch = "Done"
        self.DumpStatistics()









































#from Functions.Tools.Alerting import Notification
#from Functions.Event.DataLoader import GenerateDataLoader
#from Functions.GNN.Models import EdgeConv, InvMassGNN
#from Functions.GNN.PathNets import PathNet
#from Functions.IO.Files import WriteDirectory, Directories
#from Functions.Particles.Particles import Particle
#
#import torch
#from torch.utils.data import SubsetRandomSampler
#from torch_geometric.loader import DataLoader
#from torch_geometric.utils import accuracy
#
#import numpy as np
#from sklearn.model_selection import KFold 
#
#import time 
#
#
#class Optimizer(Notification):
#
#    def __init__(self, Loader, Debug = False):
#        self.Verbose = True
#        Notification.__init__(self, self.Verbose)
#        self.Caller = "OPTIMIZER"
#
#        self.DataLoader = {}
#        if isinstance(Loader, dict):
#            self.DataLoader = Loader
#        elif isinstance(Loader, GenerateDataLoader) and Loader.Converted == True:
#            self.DataLoader = Loader.DataLoader
#        elif isinstance(Loader, GenerateDataLoader):
#            self.DataLoader = Loader
#            self.DataLoader = self.Loader.DataLoader
#        else:
#            self.Warning("FAILED TO INITIALIZE OPTIMIZER. WRONG DATALOADER")
#            return False
#
#        self.Loader = Loader 
#        self.Epochs = 50
#        if Debug == False:
#            self.Device_s = "cuda"
#            self.Device = torch.device("cuda")
#        else:
#            self.Device = torch.device("cpu")
#            self.Device_s = "cpu"
#
#        self.Debug = Debug
#        self.LearningRate = 1e-2
#        self.WeightDecay = 1e-4
#        self.DefaultBatchSize = 25
#        self.MinimumEvents = 250
#        self.kFold = 10
#        self.NotifyTime = 5
#        self.DefaultTargetType = "Nodes"
#        self.DefaultLossFunction = ""
#        self.TrainingName = "UNTITLED"
#        self.ModelOutdir = "Models"
#        self.Model = None
#
#        self.LossTrainStatistics = {}
#        self.TrainStatistics = {}
#        
#        self.ValidationStatistics = {}
#        self.LossValidationStatistics = {}
#
#        self.BatchRate = []
#        self.BatchTime = []
#
#        self.epoch = 0
#        self.Trained = False
#
#    def kFoldTraining(self):
#        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state = 42)
#        
#        for i in self.DataLoader:
#            for k in self.DataLoader[i]:
#                k.to(self.Device_s)
#
#        MaxNode = []
#        for i in self.DataLoader:
#            if self.MinimumEvents <= len(self.DataLoader[i]):
#                MaxNode.append(int(i))
#        MaxNode.sort(reverse = True)
#        WriteDirectory().MakeDir("Models/" + self.TrainingName)
#        
#        TimeStart = time.time()
#        for epoch in range(self.Epochs):
#            self.Notify("EPOCH =============================== " +str(epoch+1) + "/" + str(self.Epochs))
#            self.epoch = epoch+1
#
#            self.LossTrainStatistics[self.epoch] = []
#            self.LossValidationStatistics[self.epoch] = []
#
#            self.ValidationStatistics[self.epoch] = []
#            self.TrainStatistics[self.epoch] = []
#            TimeStartEpoch = time.time()
#
#            for n_node in MaxNode:
#                if n_node == 0:
#                    continue
#                self.Notify("NUMBER OF NODES -----> " + str(n_node) + " BEING TESTED")
#                CurrentData = self.DataLoader[n_node] 
#                
#                if len(CurrentData) < self.kFold:
#                    self.Warning("NOT ENOUGH SAMPLES FOR EVENTS WITH " + str(n_node) + " PARTICLES :: SKIPPING")
#                    continue
#
#                for fold, (train_idx, val_idx) in enumerate(Splits.split(np.arange(len(CurrentData)))):
#                    self.Notify("CURRENT k-Fold: " + str(fold+1))
#                    
#                    train_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = SubsetRandomSampler(train_idx))
#                    valid_loader = DataLoader(CurrentData, batch_size = self.DefaultBatchSize, sampler = SubsetRandomSampler(val_idx)) 
#                    self.Notify("-------> Training <-------")
#                    self.SampleLoop("Training", train_loader)                    
#
#                    self.Notify("-------> Validation <-------")
#                    self.SampleLoop("Validation", valid_loader)
#                
#                self.Notify("CURRENT TRAIN LOSS FUNCTION: " + str(round(float(self.LossTrainStatistics[self.epoch][-1][-1]), 7)))
#                self.Notify("CURRENT ACCURACY (Validation): " + str(round(float(self.ValidationStatistics[self.epoch][-1]), 7)))
#            
#            torch.save(self.Model, self.ModelOutdir + "/" + self.TrainingName + "/Model_epoch" + str(epoch +1) +".pt")
#            TimeEndEpoch = time.time()
#            self.BatchTime.append(TimeEndEpoch - TimeStartEpoch)
#        
#        TimeEnd = time.time()
#        self.TrainingTime = TimeEnd - TimeStart
#        self.Loader.Trained = True 
#
#    def SampleLoop(self, Mode, sample_loader):
#        Mode_b = True 
#        if Mode != "Training":
#            Mode_b = False
#
#        self.ResetAll()
#        TT, TP, L = [], [], []
#        self.len = len(sample_loader)*self.DefaultBatchSize
#        for ts in sample_loader:
#            self.sample = ts
#            self.TrainClassification(Mode_b)
#            
#            for i in range(self.DefaultBatchSize):
#                self.ProgressInformation(Mode)
#
#            TT.append(self.__Truth)
#            TP.append(self.__Pred) 
#            L.append(float(self.L))
#
#            if self.Rate != -1:
#                self.BatchRate.append(self.Rate)
#
#        ac = accuracy(torch.cat(TP, dim = 0), torch.cat(TT, dim = 0))
#        if Mode_b:
#            self.TrainStatistics[self.epoch].append(ac)
#            self.LossTrainStatistics[self.epoch].append(L)
#        else:
#            self.ValidationStatistics[self.epoch].append(ac)
#            self.LossValidationStatistics[self.epoch].append(L)
#
#    def TrainClassification(self, Train = True):
#        if Train:
#            self.Model.train()
#            self.Optimizer.zero_grad()
#        else:
#            self.Model.eval()
#        
#        param, P, truth = self.TargetDefinition(self.sample, self.DefaultTargetType)
#        self.__Pred = P
#        self.__Truth = truth
#        self.L = self.LossFunction(param, truth)
#
#        if Train:
#            self.L.backward()
#            self.Optimizer.step()
#
#    def TargetDefinition(self, Sample, TargetType = "Nodes"):
#        pred = self.Model(Sample)
#        if TargetType == "Nodes":
#            _, p = pred.max(1)
#            return pred, p, Sample.y.t().contiguous().squeeze()
#
#        if TargetType == "NodeEdges":
#            edge_index = Sample.edge_index
#            p = self.Model.Adj_M[edge_index[0], edge_index[1]]
#            return p, torch.round(p).to(torch.int), torch.tensor(Sample.y[edge_index[0]] == Sample.y[edge_index[1]], dtype= torch.float).t()[0]
#
#        if TargetType == "Edges":
#            edge_index = Sample.edge_index
#            p = self.Model.Adj_M[edge_index[0], edge_index[1]]
#            return p, torch.round(p).to(torch.int), Sample.edge_y.t()[0].to(torch.float)
#
#    def DefineLossFunction(self, LossFunction):
#        if LossFunction == "CrossEntropyLoss":
#            self.LossFunction = torch.nn.CrossEntropyLoss()
#        elif LossFunction == "MSELoss":
#            self.LossFunction = torch.nn.MSELoss()
#        self.DefaultLossFunction = LossFunction
#
#    def DefineOptimizer(self):
#        self.Model.to(self.Device)
#        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
#    
#    def DefineEdgeConv(self, in_channels, out_channels):
#        self.Classifier = True
#        self.Model = EdgeConv(in_channels, out_channels)
#        self.DefineOptimizer()
#        self.DefineLossFunction("CrossEntropyLoss")
#
#    def DefineInvMass(self, out_channels, Target = "Nodes"):
#        self.Classifier = True
#        self.Model = InvMassGNN(out_channels)
#        self.DefineOptimizer()
#
#        self.DefaultTargetType = Target
#        if Target == "Nodes":
#            self.DefineLossFunction("CrossEntropyLoss")
#        elif Target == "NodeEdges": 
#            self.DefineLossFunction("MSELoss")
#        elif Target == "Edges": 
#            self.DefineLossFunction("MSELoss")
#
#    def DefinePathNet(self, complex = 64, path = 64, hidden = 64, out = 50, Target = "Nodes"):
#        self.Classifier = True
#        self.Model = PathNet(complex, path, hidden, out, self.Debug)
#        self.DefineOptimizer()
#
#        self.DefaultTargetType = Target
#        if Target == "Nodes":
#            self.DefineLossFunction("CrossEntropyLoss")
#        elif Target == "NodeEdges": 
#            self.DefineLossFunction("MSELoss")
#        elif Target == "Edges": 
#            self.DefineLossFunction("MSELoss")
#
#    def LoadModelState(self, Path = False):
#        
#        if Path == False:
#            Path = "Models/" + self.TrainingName
#
#            D = Directories("")
#            Files = D.ListFilesInDir(Path) 
#            high = []
#            for i in Files:
#                high.append(int(i.split("epoch")[1].replace(".pt", "")))
#            Path = self.ModelOutdir + "/" + self.TrainingName + "/Model_epoch" + str(max(high)) +".pt"
#        self.Model = torch.load(Path)
#
#
#    def __RebuildClustersFromEdge(self, topo):
#
#        edge_index = self.Sample.edge_index
#        TMP = {}
#        for t, e_i, e_j in zip(topo, edge_index[0], edge_index[1]):
#            if t > 0:
#                if int(e_i) not in TMP:
#                    TMP[int(e_i)] = []
#                TMP[int(e_i)].append(int(e_j))      
#        TMP_L = []
#        for i in TMP:
#            l = TMP[i]
#            l.append(i)
#            TMP_L.append(list(set(l)))
#
#        TMP_L = [list(x) for x in set(tuple(x) for x in TMP_L)]
#
#        Output = []
#        for k in TMP_L:
#            Par = Particle(True)
#            for j in k:
#                P = Particle(True)
#                P.pt = self.Sample.pt[j]
#                P.e = self.Sample.e[j]
#                P.phi = self.Sample.phi[j]
#                P.eta = self.Sample.eta[j]
#                Par.Decay.append(P)    
#            Par.CalculateMassFromChildren()
#            Output.append(Par)
#        return Output
#
#    def __RebuildClustersFromNodes(self, topo):
#        topo = topo.tolist() 
#        TMP_L = list(set(topo))
#        TMP = {}
#        for i in TMP_L:
#            TMP[i] = Particle(True)
#        
#        for i in range(len(topo)):
#            index = topo[i]
#            Par = Particle(True)
#            Par.pt = self.Sample.pt[i]
#            Par.e = self.Sample.e[i]
#            Par.phi = self.Sample.phi[i]
#            Par.eta = self.Sample.eta[i]
#            TMP[index].Decay.append(Par)
#        
#        Output = []
#        for i in TMP:
#            TMP[i].CalculateMassFromChildren()
#            Output.append(TMP[i]) 
#        return Output
#    
#    def __RebuildParticlesFromData(self, Prediction, DataLoader):
#        Output = {}
#        self.ResetAll()
#        l = 0
#        for i in DataLoader:
#            l += len(DataLoader[i]) 
#        self.len = l
#
#        for nodes in DataLoader:
#            for event in DataLoader[nodes]:
#                self.Sample = event
#                if Prediction: 
#                    _, topo, truth = self.TargetDefinition(event, self.DefaultTargetType)
#                else:
#                    if self.DefaultTargetType == "Edges":
#                        topo = event.edge_y
#                    elif self.DefaultTargetType == "Nodes":
#                        topo = event.y.t()[0]
#
#                self.ProgressInformation("BUILDING/PREDICTING")
#                if self.DefaultTargetType == "Edges":
#                    Output[event.i] = self.__RebuildClustersFromEdge(topo) 
#                elif self.DefaultTargetType == "Nodes":
#                    Output[event.i] = self.__RebuildClustersFromNodes(topo)
#        if len(Output) == 0:
#            self.Warning("NO DATA!")
#        return Output
#
#    def RebuildTruthParticles(self):
#        self.Notify("REBUILDING TRUTH PARTICLES FROM TRAINING")
#        return self.__RebuildParticlesFromData(False, self.Loader.DataLoader)
#
#    def RebuildPredictionParticles(self):
#        self.Notify("REBUILDING PREDICTION PARTICLES FROM TRAINING")
#        return self.__RebuildParticlesFromData(True, self.Loader.DataLoader) 
#
#    def RebuildTruthParticlesTestData(self):
#        self.Notify("REBUILDING TRUTH PARTICLES FROM TEST DATA")
#        return self.__RebuildParticlesFromData(False, self.Loader.TestDataLoader)
#
#    def RebuildPredictionParticlesTestData(self):
#        self.Notify("REBUILDING PREDICTION PARTICLES FROM TEST DATA")
#        return self.__RebuildParticlesFromData(True, self.Loader.TestDataLoader) 
#
#    def RebuildEventFileMapping(self):
#        it = 0 
#        File = ""
#        Mapping = {}
#        for i in self.Loader.Bundles:
#            start = i[2]
#            end = i[3]
#            Events = i[1].Events
#            for k in Events:
#                File = i[1].EventIndexFileLookup(k)
#                Mapping[it] = File
#                it += 1
#        return Mapping
