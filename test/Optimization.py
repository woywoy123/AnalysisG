from AnalysisTopGNN.Tools import Tools, RandomSamplers
from Model import Model
import torch


class Scheduler:

    def __init__(self):
        pass

    
class Optimizer:
    
    def __init__(self, model, keyword, parameters):
        self.optimizer = None
        self.optimizer = self.ADAM(model, parameters) if keyword == "ADAM" else self.optimizer
        self.optimizer = self.StochasticGradientDescent(model, parameters) if keyword == "SGD" else self.optimizer
        self.name = keyword
        self.params = parameters

    def ADAM(self, model, params):
        return torch.optim.Adam(model.parameters(), **params)

    def StochasticGradientDescent(self, model, params):
        return torch.optim.SGD(model.parameters(), **params)
    
    def __call__(self, step):
        if step:
            self.optimizer.step() 
        else:  
            self.optimizer.zero_grad() 


import time
from AnalysisTopGNN.Statistics import Metrics
class Epoch:

    def __init__(self, epoch):
        self.EpochTime = time.time()
        self.Epoch = epoch
        self.ModelOutputs = []
        
        self.NodeTimes = {} 
        
        self.FoldTime = {}
        self.Fold = None
        
        self.t_e = 0
        self.t_s = 0

        self.names = []

    def MakeDictionary(self, name):
        setattr(self, "Accuracy_"+name, {k[2:] : [] for k in self.ModelOutputs})
        setattr(self, "Loss_"+name, {k[2:] : [] for k in self.ModelOutputs})
        setattr(self, "TotalLoss_" + name, [])
        setattr(self, "ROC_"+name, {k[2:] : {"truth" : [], "p_score" : []} for k in self.ModelOutputs})
        self.names.append(name)

    def Collect(self, pred, truth, loss_acc, name):
        ROC = self.__dict__["ROC_" + name]
        Loss = self.__dict__["Loss_" + name]
        Accuracy = self.__dict__["Accuracy_" + name]
        TotalLoss = self.__dict__["TotalLoss_" + name]

        indx, count = pred.batch.unique(return_counts = True)
        sub_pred = [pred.subgraph(pred.batch == b) for b in indx]

        # Get the average node time 
        av_n = (self.t_e - self.t_s) / pred.num_nodes
        for i in sub_pred:
            if i.num_nodes not in self.NodeTimes:
                self.NodeTimes[i.num_nodes] = []
            self.NodeTimes[i.num_nodes].append(i.num_nodes*av_n) 
         
        # Get the loss for this prediction
        self.TotalLoss = 0
        for key in Accuracy: 
            Accuracy[key].append(loss_acc[key][1].detach().cpu().item())
            self.TotalLoss += loss_acc[key][0]        
            Loss[key].append(loss_acc[key][0].detach().cpu().item())
        TotalLoss += [self.TotalLoss.detach().cpu().item()]

        tru = truth.to_dict()
        pred = pred.to_dict()
        for key in ROC:
            ROC[key]["truth"] += tru[key].view(-1).detach().cpu().tolist()
            ROC[key]["p_score"] += pred[key].softmax(dim = 1).max(1)[0].view(-1).detach().cpu().tolist()

        self.FoldTime[self.Fold] += (self.t_e - self.t_s)
    
    def StartTimer(self):
        self.t_s = time.time()

    def StopTimer(self):
        self.t_e = time.time()
    
    def Process(self, DoStats = True):
        self.EpochTime = time.time() - self.EpochTime
        self.TotalLoss = None
        self.DoStatistics = DoStats



class ModelTrainer(Tools, RandomSamplers):

    def __init__(self):
        self.Model = None
        self.Device = None

        self.Epochs = 10
        self.kFolds = 10

        self.Scheduler = None
        self.Optimizer = None
        self.BatchSize = 20

        self.Samples = {}
        self.Tree = None
        self.SplitSampleByNode = False

        self.DoStatistics = True

    def AddAnalysis(self, analysis):
        for smpl in analysis:
            if smpl.Train == False:
                continue
            if smpl.Compiled == False:
                continue
            key = "All"
            if self.SplitSampleByNode:
                key = smpl.Trees[self.Tree].num_nodes.item()

            if key not in self.Samples:
                self.Samples[key] = []
            smpl.Trees[self.Tree].to(self.Device)

            self.Samples[key].append(smpl.Trees[self.Tree])
            

    def kFoldTraining(self, sample):
        
        folds = self.MakekFolds(sample, self.kFolds, self.BatchSize) 
        if folds == None:
            return 

        for f in folds:
            train = folds[f][0]
            validation = folds[f][1]
            
            self.Model.train()
            self._EpochObj.Fold = f
            self._EpochObj.FoldTime[f] = 0
            for i in train:
                self.Optimizer(False) 
                
                self._EpochObj.StartTimer()
                pred, truth, loss_acc = self.Model.Prediction(i) 
                self._EpochObj.StopTimer()

                self._EpochObj.Collect(pred, truth, loss_acc, "Train")

                loss = self._EpochObj.TotalLoss
                loss.backward()
                self.Optimizer(True)
            
            self.Model.eval()
            for i in validation:
                self._EpochObj.StartTimer()
                pred, truth, loss_acc = self.Model.Prediction(i) 
                self._EpochObj.StopTimer()

                self._EpochObj.Collect(pred, truth, loss_acc, "Validation")


    def Train(self):
        
        for epoch in range(self.Epochs):
            self._EpochObj = Epoch(epoch)
            self._EpochObj.ModelOutputs += list(self.Model.GetModelOutputs())
            self._EpochObj.MakeDictionary("Train")
            self._EpochObj.MakeDictionary("Validation")
            for nodes in self.Samples:
                print("-->", nodes)
                self.kFoldTraining(self.Samples[nodes])
            self._EpochObj.Process(self.DoStatistics)
            print(self._EpochObj.__dict__)



    def Launch(self):
        sample = list(self.Samples.values())[0][-1]
        optim = list(self.Optimizer)[0] 
        self.Optimizer = Optimizer(self.Model, optim, self.Optimizer[optim])
        
        self.Model = Model(self.Model)
        self.Model.Device = self.Device
        train, pred = self.Model.SampleCompatibility(sample)
            
        # Add condition to turn model into eval mode if pred != train
        self.Train()
