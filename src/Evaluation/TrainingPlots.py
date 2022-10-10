from Tooling import Template
from LogDump import LogDumper

from AnalysisTopGNN.Plotting.TemplateLines import TLine


class Training(Template, LogDumper):
    
    def __init__(self):
        self._S = " | "
        self.ValidLoss = {}
        self.ValidAcc = {}

        self.TrainLoss = {}
        self.TrainAcc = {}
        
        self.Epochs = []
        self.EpochTime = []
        self.kFolds = []
        self.FoldTime = []

        self.NodeValidLoss = {}
        self.NodeValidAcc = {}

        self.NodeTrainLoss = {}
        self.NodeTrainAcc = {}

        self.Plots = {}

    def AddEpoch(self, epoch, dic):
        self.AddPoint(dic["TrainingLoss"], self.TrainLoss, epoch)
        self.AddPoint(dic["TrainingAccuracy"], self.TrainAcc, epoch)
        self.AddPoint(dic["ValidationLoss"], self.ValidLoss, epoch)
        self.AddPoint(dic["ValidationAccuracy"], self.ValidAcc, epoch)
        self.Epochs.append(epoch)

        self.EpochTime.append(dic["EpochTime"])
        self.FoldTime += dic["FoldTime"]
        self.kFolds += dic["KFolds"]

        self.AddPoint(dic["NodeTrainingLoss"], self.NodeTrainLoss, epoch)
        self.AddPoint(dic["NodeTrainingAccuracy"], self.NodeTrainAcc, epoch)
        self.AddPoint(dic["NodeValidationLoss"], self.NodeValidLoss, epoch)
        self.AddPoint(dic["NodeValidationAccuracy"], self.NodeValidAcc, epoch)
   
    def Compile(self, Output):
        self.Plots["TrainingLoss"] = {}
        self.Plots["ValidationLoss"] = {}
        self.Plots["TrainingAccuracy"] = {}
        self.Plots["ValidationAccuracy"] = {}
        ToReturn = {} 
        self.OutDir = Output + "/training/plots/"
        for feat in self.TrainLoss:
            self.MakeLossPlot(self.TrainLoss, feat, "Training", self.OutDir + "loss", "-")
            self.MakeLossPlot(self.ValidLoss, feat, "Validation", self.OutDir + "loss", "--")
            
            self.Plots["TrainingLoss"] |= self.TrainLoss
            self.Plots["ValidationLoss"] |= self.ValidLoss

            comb = self.MergePlots([self.TrainLoss[feat], self.ValidLoss[feat]], self.OutDir + "loss")
            comb.Title = "Loss for Feature: " + feat
            comb.Filename = "Loss_" + feat
            comb.SaveFigure()
            
            self.MakeAccuracyPlot(self.TrainAcc, feat, "Training", self.OutDir + "accuracy", "-")
            self.MakeAccuracyPlot(self.ValidAcc, feat, "Validation", self.OutDir + "accuracy", "--")

            self.Plots["TrainingAccuracy"] |= self.TrainAcc
            self.Plots["ValidationAccuracy"] |= self.ValidAcc

            comb = self.MergePlots([self.TrainAcc[feat], self.ValidAcc[feat]], self.OutDir + "accuracy")
            comb.Title = "Accuracy for Feature: " + feat
            comb.Filename = "Accuracy_" + feat
            comb.SaveFigure()


        loss, acc = {}, {}
        for node in self.NodeValidLoss:
            
            self.MakeLossPlot(self.NodeValidLoss, node, "Validation", self.OutDir + "loss", "-")
            self.MakeLossPlot(self.NodeTrainLoss, node, "Training", self.OutDir + "loss", "--")
            
            self.MakeAccuracyPlot(self.NodeValidAcc, node, "Validation", self.OutDir + "accuracy", "-")
            self.MakeAccuracyPlot(self.NodeTrainAcc, node, "Training", self.OutDir + "accuracy", "--")

            n = node.split("/")
            nd = n[0]
            feat = n[1]
            
            if feat not in loss:
                loss[feat] = []
                acc[feat] = []

            self.NodeValidLoss[node].Title = "Valid - node - " + nd
            self.NodeTrainLoss[node].Title = "Train - node - " + nd
            self.NodeValidLoss[node].LineStyleMarker = "-" 
            self.NodeTrainLoss[node].LineStyleMarker = "--" 
            for i in range(int(nd)):
                self.NodeValidLoss[node].ApplyRandomColor(self.NodeValidLoss[node])
            self.NodeTrainLoss[node].Color = self.NodeValidLoss[node].Color
            loss[feat].append(self.NodeValidLoss[node])
            loss[feat].append(self.NodeTrainLoss[node])


            self.NodeValidAcc[node].Title = "Valid - node - " + nd
            self.NodeTrainAcc[node].Title = "Train - node - " + nd
            self.NodeValidAcc[node].LineStyle = "-" 
            self.NodeTrainAcc[node].LineStyle = "--" 
            for i in range(int(nd)):
                self.NodeValidAcc[node].ApplyRandomColor(self.NodeValidAcc[node])
            self.NodeTrainAcc[node].Color = self.NodeValidAcc[node].Color
            acc[feat].append(self.NodeValidAcc[node])
            acc[feat].append(self.NodeTrainAcc[node])

        for feat in loss:
            Comb = self.MergePlots(loss[feat], self.OutDir + "loss")
            Comb.Filename = "NodeFeatureLoss_" + feat
            Comb.Title = "Loss of Feature " + feat
            Comb.SaveFigure()

            Comb = self.MergePlots(acc[feat], self.OutDir + "accuracy")
            Comb.Filename = "NodeFeatureAccuracy_" + feat
            Comb.Title = "Accuracy of Feature " + feat
            Comb.SaveFigure()
 

        ToWrite = {}
        for feat in self.Plots["TrainingLoss"]:
            m = ["TrainingLoss", "ValidationLoss",  "TrainingAccuracy", "ValidationAccuracy"]
            t = ["Training Loss", "Validation Loss",  "Training Accuracy", "Validation Accuracy"]
            u = ["a.u", "a.u","%", "%"]
            dump = []
            for j, k, f in zip(m, t, u):
                self.Plots[j][feat].Title = k
                self.Plots[j][feat].yTitle = f
                dump.append(self.Plots[j][feat])
            ToWrite[feat] = self.DumpTLines(dump) 
        
        for i in ToWrite:
            self.WriteText(ToWrite[i], self.OutDir + "/" + i)
        
        EpochT = {}
        for ep in range(len(self.Epochs)):
            EpochT[self.Epochs[ep]] = self.EpochTime[ep]
        self.EpochTime = EpochT
        self.SortEpoch(self.EpochTime)

        Plot = self.TemplateTLine(Output + "/training/plots/time")
        Plot["xData"] = list(self.EpochTime)
        Plot["yData"] = list(self.EpochTime.values())
        Plot["xTitle"] = "Epoch"
        Plot["yTitle"] = "Time (s)"
        Plot["Filename"] = "EpochTime"
        Plot["Title"] = "Time Spent on each Epoch"
        TL = TLine(**Plot)
        TL.SaveFigure()
        self.Plots["EpochTime"] = TL
        ToReturn["EpochTime"] = TL

        merge = []
        for feat in self.TrainAcc:
            self.TrainAcc[feat].Title = feat
            self.TrainAcc[feat].Marker = None
            self.TrainAcc[feat].Color = None
            merge.append(self.TrainAcc[feat])

        Comb = self.MergePlots(merge, self.OutDir + "accuracy")
        Comb.Filename = "MergedFeatureTrainingAccuracy"
        Comb.Title = "Accuracy for All Features - Training"
        Comb.SaveFigure()
        ToReturn["Accuracy"] = Comb

        return ToReturn 
