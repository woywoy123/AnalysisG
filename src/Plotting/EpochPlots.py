from .TemplateHistograms import TH1F, CombineTH1F
from .TemplateLines import TLine, CombineTLine
from AnalysisTopGNN.Statistics import Metrics

class Container(object):
    def __init__(self):
        self.yData = []
        self.errData = []
        self.FeatureContainer = {}
        self.Plots = []

    def Add(self, data):
        if isinstance(data, dict):
            incoming = {k : Container() for k in data if k not in self.FeatureContainer}
            self.FeatureContainer |= incoming
            for i in data:
                self.FeatureContainer[i].Add(data[i])
            return 
        if isinstance(data, float):
            self.yData += [data]
            return 

        self.yData += [data[0]]
        self.errData += [data[1]]

class EpochPlots:

    def __init__(self):
        pass
    
    def NodeTimingHistograms(self):
        Outdir = self.OutDir + "/Epoch-"+ str(self.Epoch) + "/DebugPlots"
        Plots = {"yTitle" : "Entries", "xTitle" : "Seconds", "xBins" : 1000, "xMin" : 0}
        Hists = []
        for n in self.NodeTimes:
            Plots["xData"] = self.NodeTimes[n]
            Plots["Title"] = "Node-" + str(n)
            H = TH1F(**Plots)
            Hists.append(H)
        Plots["OutputDirectory"] = Outdir 
        Plots["Histograms"] = Hists
        Plots["Title"] = "Different Node Size Time Distributions"
        Plots["Filename"] = "NodeSizeTiminingDistribution"
        T = CombineTH1F(**Plots)
        T.SaveFigure()
    
    def CompileROC(self, Names):
        Plots = {
                    "yTitle" : "True Positive Rate", 
                    "xTitle" : "False Positive Rate", 
                    "xMin" : 0, "xMax" : 1, 
                    "yMin" : 0, "yMax" : 1, 
                }
        
        for feat in [i[2:] for i in self.ModelOutputs]:
            Plots["OutputDirectory"] = self.OutDir + "/Epoch-" + str(self.Epoch) + "/ROC"
            line = []
            for name in self.names:
                ROC = getattr(self, "ROC_" + name)
                ROC[feat] = Metrics().MakeROC(ROC[feat]["truth"], ROC[feat]["p_score"])
                Plots["yData"] = ROC[feat]["tpr"]
                Plots["xData"] = ROC[feat]["fpr"]
                Plots["Title"] = name
                L = TLine(**Plots)
                L.Compile()
                line.append(L)
            Plots["Title"] = "Receiver and Operator Curve: " + feat
            Plots["Lines"] = line
            Plots["Filename"] = feat
            C = CombineTLine(**Plots)
            C.SaveFigure()

    def AccuracyHistograms(self):
        for feat in [i[2:] for i in self.ModelOutputs]:
            Outdir = self.OutDir + "/Epoch-"+ str(self.Epoch) + "/DebugPlots"
            Plots = {
                        "yTitle" : "Entries", 
                        "xTitle" : "Accuracy (%)", 
                        "xBins" : 100, 
                        "xMin" : 0, "xMax" : 100, 
                        "Filename" : feat + "_Accuracy", "OutputDirectory" : Outdir
                    }

            Hists = []
            for names in self.names:
                Acc = getattr(self, "Accuracy_" + names)[feat]
                Plots["Title"] = names
                Plots["xData"] = [100*k for k in Acc]
                Hists.append(TH1F(**Plots))    
            Plots["Title"] = "Accuracy Distribution for Different Samples Feature: " + feat
            Plots["Histograms"] = Hists
            CH = CombineTH1F(**Plots)
            CH.SaveFigure()

    def LossHistograms(self):
        for feat in [i[2:] for i in self.ModelOutputs]:
            Outdir = self.OutDir + "/Epoch-"+ str(self.Epoch) + "/DebugPlots"
            Plots = {
                        "yTitle" : "Entries", 
                        "xTitle" : "Loss of Prediction", 
                        "xBins" : 100, "xMin" : 0,  
                        "Filename" : feat + "_Loss", "OutputDirectory" : Outdir
                    }

            Hists = []
            for names in self.names:
                Acc = getattr(self, "Loss_" + names)[feat]
                Plots["Title"] = names
                Plots["xData"] = Acc
                Hists.append(TH1F(**Plots))    
            Plots["Title"] = "Loss Distribution for Different Samples Feature: " + feat
            Plots["Histograms"] = Hists
            CH = CombineTH1F(**Plots)
            CH.SaveFigure()


        Outdir = self.OutDir + "/Epoch-"+ str(self.Epoch) + "/DebugPlots"
        Plots = {
                    "yTitle" : "Entries", 
                    "xTitle" : "Loss of Prediction", 
                    "xBins" : 100, "xMin" : 0, 
                    "Filename" : "TotalLoss", "OutputDirectory" : Outdir
                }
        
        Hists = []
        for names in self.names:
            Loss = getattr(self, "TotalLoss_" + names)
            Plots["Title"] = names
            Plots["xData"] = Loss
            Hists.append(TH1F(**Plots))    
        Plots["Title"] = "Total Loss Distribution for Different Samples Feature"
        Plots["Histograms"] = Hists
        CH = CombineTH1F(**Plots)
        CH.SaveFigure()


    
    def TemplateLine(self, xData, yData, outdir, feature, up, down):
        Plots = {
                    "xTitle" : "Epoch",
                    "xData" : xData, 
                    "yData" : yData, 
                    "up_yData" : up, 
                    "down_yData" : down, 
                    "Title" : feature,
                    "yMin" : 0,
                    "yMax" : None if len(up) == 0 else max([k+i for k, i in zip(up, yData)]),
                    "Style" : "ATLAS",
                    "OutputDirectory" : outdir 
                }
        return Plots
    
    def AccuracyPlot(self, xData, yData, outdir, mode, feature, up, down):
        Plots = self.TemplateLine(xData, [i*100 for i in yData], outdir + "/Accuracy", mode, up, down)
        Plots |= {"yTitle" : "Accuracy (%)"}
        return TLine(**Plots)

    def EpochTimePlot(self, xData, yData, outdir):
        Plots = self.TemplateLine(xData, yData, outdir + "/Time", "Time", [], [])
        Plots |= {"yTitle" : "Seconds"}
        Plots |= {"Filename" : "EpochTime"}
        Plots |= {"Title" : "Time Elapsed per Epoch"}
        return TLine(**Plots)

    def FoldTimePlot(self, xData, yData, outdir, up, down):
        Plots = self.TemplateLine(xData, yData, outdir + "/Time", "Time", up, down)
        Plots |= {"yTitle" : "Average Time Spent Folding (seconds)"}
        Plots |= {"Filename" : "FoldTime"}
        Plots |= {"Title" : "Average Folding Time Elapsed per Epoch"}
        return TLine(**Plots)

    def NodeTimePlot(self, xData, yData, outdir, Nodes, up, down):
        Plots = self.TemplateLine(xData, yData, outdir + "/Time", Nodes, up, down)
        Plots |= {"yTitle" : "Seconds"}
        plt = TLine(**Plots) 
        plt.Compile()
        return plt

    def AUCPlot(self, xData, yData, outdir, mode):
        Plots = self.TemplateLine(xData, yData, outdir + "/AUC", mode, [], [])
        Plots |= {"yTitle" : "Area under ROC Curve (Higher is Better)"}
        return TLine(**Plots) 

    def LossPlot(self, xData, yData, outdir, mode, up, down):
        Plots = self.TemplateLine(xData, yData, outdir + "/Loss", mode, up, down)
        Plots |= {"yTitle" : "Loss of Prediction (Lower is Better) (a.u)"}
        return TLine(**Plots) 

    def GetConsistentModeColor(self, Color):
        Plt = CombineTLine()
        Line = TLine()
        for i in Color:
            Plt.ApplyRandomColor(Line)
            Color[i] = Line.Color
            Line.Color = None

    def GetConsistentModeMarker(self, Marker):
        option = [",", ".", "-", "x", "o", "O"]
        for i in Marker:
            Marker[i] = option.pop(0)

    def MakeConsistent(self, inpt, Color, Marker):
        for feat in inpt:
            col = Color[feat]
            for p in inpt[feat]:
                p.Plots.Marker = Marker[p.Plots.Title]
                p.Plots.Color = col
                
                p.Plots.Title += "-" + feat
        
    def MergeNodeTimes(self, NodeTimes, outdir):
        Plots = self.TemplateLine([], [], outdir + "/Time", None, [], [])
        Plots["Filename"] = "NodeTime"
        Plots["Title"] = "Average Time Spent on n-Nodes"
        Plots["LegendOn"] = False
        Plots["Lines"] = [i.Plots for i in NodeTimes]
        Com = CombineTLine(**Plots)
        Com.SaveFigure() 

    def MergeAUC(self, AUC, Colors, Markers, outdir):
        self.MakeConsistent(AUC, Colors, Markers)
        Plots = self.TemplateLine([], [], outdir + "/AUC", None, [], [])
        Plots["yTitle"] = "Area under ROC Curve (Higher is Better)"
        Plots["Lines"] = [i.Plots for feat in AUC for i in AUC[feat]]
        Plots["Filename"] = "AUC"
        Plots["yMax"] = 1
        Com = CombineTLine(**Plots)
        Com.SaveFigure()
    
    def MergeLoss(self, Loss, Colors, Markers, outdir):
        Plots = self.TemplateLine([], [], outdir + "/Loss", None, [], [])
        Plots["Filename"] = "TotalLoss"
        Plots["Title"] = "Aggregated Loss of Model Predictions"
        Plots["Lines"] = [i.Plots for i in Loss["Total"]]
        Plots["yMax"] = max([j+k for i in Loss["Total"] for j, k in zip(i.Plots.yData, i.Plots.up_yData) ])
        Com = CombineTLine(**Plots)
        Com.SaveFigure()
        
        features = list({l : Loss[l] for l in Loss if l != "Total"})
        self.MakeConsistent({l : Loss[l] for l in features}, Colors, Markers)
        Plots = self.TemplateLine([], [], outdir + "/Loss", None, [], [])
        Plots["Filename"] = "AllFeatures"
        Plots["Title"] = "Model Prediction Loss for Features"
        Plots["Lines"] = [j.Plots for i in features for j in Loss[i]]
        Plots["yMax"] = max([j+k for f in features for i in Loss[f] for j, k in zip(i.Plots.yData, i.Plots.up_yData) ])
        Com = CombineTLine(**Plots)
        Com.SaveFigure()
    
    def MergeAccuracy(self, Accuracy, Colors, Markers, outdir):
        features = list(Accuracy)
        self.MakeConsistent(Accuracy, self.Colors, self.Markers)
        Plots = self.TemplateLine([], [], outdir + "/Accuracy", None, [], [])
        Plots["Filename"] = "AllFeatures"
        Plots["Title"] = "Model Prediction Accuracy for Features"
        Plots["Lines"] = [j.Plots for i in features for j in Accuracy[i]]
        Plots["yMax"] = 100
        Com = CombineTLine(**Plots)
        Com.SaveFigure()



