from AnalysisTopGNN.Tools import Tools, RandomSamplers
from AnalysisTopGNN.IO import PickleObject
from AnalysisTopGNN.Samples import Epoch
from AnalysisTopGNN.Model import Model, Optimizer, Scheduler
from AnalysisTopGNN.Notification import Optimization
from AnalysisTopGNN.Samples import SampleTracer
from .Settings import Settings


class Optimization(Tools, RandomSamplers, Optimization, Settings):

    def __init__(self):
        Settings.__init__(self)
        self.Settings = Settings()
        self.Caller = "Optimization"
        
        self.Samples = {}
        

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
            
            self.StartKfoldInfo(train, validation, f, len(folds))

            self.Model.train()
            self._EpochObj.Fold = f
            
            if f not in self._EpochObj.FoldTime:
                self._EpochObj.FoldTime[f] = 0

            self.Model.train()
            for i in train:
                
                self._EpochObj.StartTimer()
                pred, truth, loss_acc = self.Model.Prediction(i) 
                self._EpochObj.StopTimer()
                self._EpochObj.Collect(pred, truth, loss_acc, "Train")
                loss = self._EpochObj.TotalLoss
                
                self.TrainingInfo(len(train), self._EpochObj, pred, truth, loss_acc, self.DebugMode)

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
            self.ShowEpoch(epoch, self.Epochs)
            outputDir = self.ProjectName + "/TrainedModels/" + self.RunName + "/Epoch-"+ str(epoch)
            
            if self.IsFile(outputDir + "/TorchSave.pth") and self.ContinueTraining:
                self.Optimizer.LoadState(outputDir)
                self.Model.LoadModel(outputDir + "/TorchSave.pth")
                continue
            else:
                self.ContinueTraining = False

            self._EpochObj = Epoch(epoch)
            self._EpochObj.ModelOutputs += list(self.Model.GetModelOutputs())
            self._EpochObj.MakeDictionary("Train")
            self._EpochObj.MakeDictionary("Validation")
            self._EpochObj.OutDir = self.ProjectName + "/TrainedModels/" + self.RunName
            
            for nodes in self.Samples:
                self.TrainingNodes(nodes)
                self.kFoldTraining(self.Samples[nodes])
            
            self._EpochObj.Process()

            self.Optimizer.DumpState(outputDir)
            self.Model.DumpModel(outputDir)

            PickleObject(self._EpochObj, outputDir + "/Stats") 
            
            if self.Scheduler != None:
                self.Scheduler()
            
    def Launch(self):
        self.CheckGivenSample(self.Samples)
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
