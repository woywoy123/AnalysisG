import torch
from torch_geometric.utils import accuracy
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.Plotting.Histograms import TH1F, TH2F, TGraph, CombineTGraph
from Functions.Tools.Alerting import Notification
from Functions.GNN.Optimizer import Optimizer

class EvaluationMetrics(Notification):

    def __init__(self):
        Notification.__init__(self)
        self.Sample = []
        self.__TruthAttribute = []
        self.__PredictionAttribute = []
        self.__Type = ""
        self.__LossStatistics = ""
        self.__Pairs = {}
        self.__Truth = []
        self.__Pred = []
        self.Caller = "::EvaluationMetrics"
        self.__Processed = False

    def ProcessSample(self):
        
        self.__IsGeneratorType()
        for i, j in zip(self.__TruthAttribute, self.__PredictionAttribute):
            self.__Pairs[i] = j

        Truth = {}
        Pred = {}

        for i in self.__TruthAttribute:
            Truth[i] = []
        for i in self.__PredictionAttribute:
            Pred[i] = []

        if "PROCESSED" in self.__Type:
            for i in self.Sample.DataLoader:
                for n_p in self.Sample.EventData[i]:
                    p = n_p.NodeParticleMap
                    for t_a in self.__TruthAttribute:
                        Truth[t_a].append(self.__GetParticleAttribute(p, t_a)) 
                    
                    for m_a in self.__PredictionAttribute:
                        Pred[m_a].append(self.__GetParticleAttribute(p, m_a))
            for k in Pred:
                Pred[k] = torch.tensor(Pred[k])
            for k in Truth:
                Truth[k] = torch.tensor(Truth[k])
            self.__Processed = True
        elif "TRAINED" in self.__Type:
            pass
        else:
            self.Warning("NOTHING HAS BEEN PROCESSED!")

        self.__Pred = Pred
        self.__Truth = Truth
         
    def AddTruthAttribute(self, attr):
        self.__TruthAttribute.append(attr)

    def AddPredictionAttribute(self, attr):
        self.__PredictionAttribute.append(attr)

    def Accuracy(self):
        
        if self.__Processed == False:
            self.ProcessSample()

        for tru in self.__Pairs:
            pre = self.__Pairs[tru]
            print("Accuracy (" + tru + "-" + pre +"): " + str(round(accuracy(self.__Truth[tru], self.__Pred[pre]), 3)))

    def __GetParticleAttribute(self, Particles, Attribute):
        out = [] 
        for i in Particles:
            p = Particles[i]
            out.append(getattr(p, Attribute))
        return out

    def LossTrainingPlot(self, Dir, ErrorBars = False):
        self.__Plots = {}
        
        LossVal = self.__CompilePlot(self.__LossValidationStatistics, "Loss", "Epoch")
        LossVal.Color = "blue"
        LossVal.Title = "Validation"
        LossVal.ErrorBars = ErrorBars
        
        LossTra = self.__CompilePlot(self.__LossTrainStatistics, "Loss", "Epoch")
        LossTra.Color = "Orange"
        LossTra.Title = "Training"
        LossTra.ErrorBars = ErrorBars

        ACVal = self.__CompilePlot(self.__ValidationStatistics, "Accuracy", "Epoch")
        ACVal.Color = "blue"
        ACVal.Title = "Validation"
        ACVal.ErrorBars = ErrorBars
        ACTra = self.__CompilePlot(self.__TrainStatistics, "Accuracy", "Epoch")
        ACTra.Color = "Orange"
        ACTra.Title = "Training"
        ACTra.ErrorBars = ErrorBars
        
        C = CombineTGraph()
        C.Title = "Training and Validation Prediction Accuracy with Epoch"
        C.yMin = 0
        C.yMax = max([max(ACVal.yData), max(ACTra.yData)])*2
        C.Lines = [ACVal, ACTra] 
        C.CompileLine()
        C.Save(Dir)

        T = CombineTGraph()
        T.Title = "Training and Validation Loss Function with Epoch"
        T.yMin = 0
        T.yMax = max([max(LossVal.yData), max(LossTra.yData)])*2
        T.Lines = [LossVal, LossTra] 
        T.CompileLine()
        T.Save(Dir)

    
    def __CompilePlot(self, dic, yTitle, xTitle):
        
        T = TGraph()
        Epoch = []
        Loss = []
        for i in dic:
            l = dic[i]
            x = []
            for k in l:
                if isinstance(k, list):
                    for j in k:
                        x.append(j.item())
                else:
                    x.append(k)
            Epoch.append(i)
            Loss.append(x)
        T.xData = Epoch
        T.yData = Loss
        T.yTitle = yTitle
        T.xTitle = xTitle
        T.Line()
        return T
        

    def __IsGeneratorType(self):
        if isinstance(self.Sample, GenerateDataLoader):
            self.__Type += "DATALOADER"
            if self.Sample.Converted == True:
                self.__Type += "|CONVERTED"

            if self.Sample.TrainingTestSplit == True:
                self.__Type += "|TESTSAMPLE"

            if self.Sample.Processed == True:
                self.__Type += "|PROCESSED"
        
        if isinstance(self.Sample, Optimizer):
            if self.Sample.Loader.Trained == True:
                self.__Type += "|TRAINED"
                self.__LossValidationStatistics = self.Sample.LossValidationStatistics
                self.__LossTrainStatistics = self.Sample.LossTrainStatistics
                self.__TrainStatistics = self.Sample.TrainStatistics
                self.__ValidationStatistics = self.Sample.ValidationStatistics

    def Reset(self):
        self.__init__(self)
        

