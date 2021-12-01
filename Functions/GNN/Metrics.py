import torch
from torch_geometric.utils import accuracy
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.Plotting.Histograms import TH1F, TH2F

class EvaluationMetrics:

    def __init__(self):
        self.Sample = []
        self.__TruthAttribute = []
        self.__PredictionAttribute = []
        self.__Type = ""
        self.__LossStatistics = ""
        self.__Pairs = {}
        self.__Truth = []
        self.__Pred = []

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
            
        self.__Pred = Pred
        self.__Truth = Truth
    
    def AddTruthAttribute(self, attr):
        self.__TruthAttribute.append(attr)

    def AddPredictionAttribute(self, attr):
        self.__PredictionAttribute.append(attr)

    def Accuracy(self):
        for tru in self.__Pairs:
            pre = self.__Pairs[tru]
            print("Accuracy (" + tru + "-" + pre +"): " + str(round(accuracy(self.__Truth[tru], self.__Pred[pre]), 3)))

    def __GetParticleAttribute(self, Particles, Attribute):
        out = [] 
        for i in Particles:
            p = Particles[i]
            out.append(getattr(p, Attribute))
        return out

    def __IsGeneratorType(self):
        if isinstance(self.Sample, GenerateDataLoader):
            self.__Type += "DATALOADER"
            if self.Sample.Converted == True:
                self.__Type += "|CONVERTED"

            if self.Sample.TrainingTestSplit == True:
                self.__Type += "|TESTSAMPLE"

            if self.Sample.Processed == True:
                self.__Type += "|PROCESSED"

            if self.Sample.Trained == True:
                self.__Type += "|TRAINED"
                self.LossValidationStatistics = self.Sample.LossValidationStatistics
                self.LossTrainStatistics = self.Sample.LossTrainStatistics
                self.TrainStatistics = self.Sample.TrainStatistics
                self.ValidationStatistics = self.Sample.ValidationStatistics

    def Reset(self):
        self.__init__(self)
        

