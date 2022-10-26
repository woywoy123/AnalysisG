from AnalysisTopGNN.Tools import Tools, RandomSamplers
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Samples import Epoch
from AnalysisTopGNN.Model import Model
import torch


from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
class Scheduler:

    def __init__(self, optimizer, keyword, parameters):
        parameters["optimizer"] = optimizer
        self.scheduler = None
        self.scheduler = self.ExponentialLR(parameters) if keyword == "ExponentialLR" else self.scheduler
        self.scheduler = self.CyclicLR(parameters) if keyword == "CyclicLR" else self.scheduler 

    def ExponentialLR(self, params):
        return ExponentialLR(**params)
    
    def CyclicLR(self, params):
        return CyclicLR(**params)
    
    def __call__(self):
        self.scheduler.step()
    
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

    def DumpState(self, OutputDir):
        PickleObject(self.optimizer.state_dict(), OutputDir + "/OptimizerState")
    
    def LoadState(self, InputDir):
        self.optimizer.load_state_dict(UnpickleObject(InputDir + "/OptimizerState"))



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

        self.ProjectName = None
        self.RuName = None

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
            smpl.Trees[self.Tree].to(self.Device, non_blocking = True)

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

            self.Model.train()
            for i in train:
                
                self._EpochObj.StartTimer()
                pred, truth, loss_acc = self.Model.Prediction(i) 
                self._EpochObj.StopTimer()
                
                self._EpochObj.Collect(pred, truth, loss_acc, "Train")

                loss = self._EpochObj.TotalLoss
                print("--->", loss)
                
                self.Optimizer(step = False) 
                loss.backward()
                self.Optimizer(step = True)
            
            self.Model.eval()
            for i in validation:
                self._EpochObj.StartTimer()
                pred, truth, loss_acc = self.Model.Prediction(i) 
                self._EpochObj.StopTimer()
                self._EpochObj.Collect(pred, truth, loss_acc, "Validation")


    def Train(self):
        
        for epoch in range(self.Epochs):
            outputDir = self.ProjectName + "/" + self.RunName + "/Epoch-"+ str(epoch)
            self._EpochObj = Epoch(epoch)
            self._EpochObj.ModelOutputs += list(self.Model.GetModelOutputs())
            self._EpochObj.MakeDictionary("Train")
            self._EpochObj.MakeDictionary("Validation")
            self._EpochObj.OutDir = self.ProjectName + "/" + self.RunName
            for nodes in self.Samples:
                print("-->", nodes)
                self.kFoldTraining(self.Samples[nodes])
            self._EpochObj.Process()

            self.Optimizer.DumpState(outputDir)
            self.Model.DumpModel(outputDir)

            PickleObject(self._EpochObj, outputDir + "/Stats") 
            
            if self.Scheduler != None:
                self.Scheduler()
            
    def Launch(self):
        sample = list(self.Samples.values())[0][-1]
        optim = list(self.Optimizer)[0] 
        self.Optimizer = Optimizer(self.Model, optim, self.Optimizer[optim])
    
        if self.Scheduler != None:
            scheduler = list(self.Scheduler)[0]
            self.Scheduler = Scheduler(self.Optimizer.optimizer, scheduler, self.Scheduler[scheduler])
        
        self.Model = Model(self.Model)
        self.Model.Device = self.Device
        train, pred = self.Model.SampleCompatibility(sample)
            
        self.Train()
