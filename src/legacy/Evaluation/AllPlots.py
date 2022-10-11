from Tooling import Template
from LogDump import LogDumper

from AnalysisTopGNN.Plotting import TLine, TH1FStack

class Common(Template, LogDumper):

    def __init__(self):
        pass        

    def AddEpoch(self, epoch, dic):
        for feat in dic["AUC/AllCollected"]:
            if feat not in self.Acc:
                self.Acc[feat] = {}
                self.Loss[feat] = {}
                self.AUC[feat] = {}
            
            self.AUC[feat][epoch] = dic["AUC/AllCollected"][feat]
            self.ROC[epoch] = dic["ROC/CombinedFeatures"]

            self.Acc[feat][epoch] = dic["Accuracy/" + feat]
            self.Loss[feat][epoch] = dic["Loss/" + feat]

        self.AddEpochEdge(dic, epoch)
        self.AddEpochNode(dic, epoch)
    
    def Compile(self, Output, Mode = "all"):
        self.Plots = {
                        "Accuracy" : None, 
                        "Loss" : None, 
                        "AUC" : None, 
                        "EdgeProcessEfficiency" : None, 
                        "EdgeEfficiency" : None, 
                        "NodeProcessEfficiency" : None, 
                        "NodeEfficiency" : None
                    }

        if len(self.Acc) == 0:
            self.Warning(Mode + " has no samples. Make the HDF5 sample larger. Skipping...")
            return self.Plots
        
        self.OutDir = Output + "/" + Mode + "/plots/"
        for i in self.Acc:
            self.MakeAccuracyPlot(self.Acc, i, i, self.OutDir, "-")
            self.MakeLossPlot(self.Loss, i, i, self.OutDir, "-")

        comb1 = self.MergePlots([self.Acc[i] for i in self.Acc], self.OutDir)
        comb1.Title = "Accuracy of Predicted Features"
        comb1.Filename = "AccuracyForAllFeatures"
        comb1.SaveFigure()
        self.Plots["Accuracy"] = comb1

        comb2 = self.MergePlots([self.Loss[i] for i in self.Loss], self.OutDir)
        comb2.Title = "Loss of Predicted Features"
        comb2.Filename = "LossForAllFeatures"
        comb2.SaveFigure()
        self.Plots["Loss"] = comb2

        self.SortEpoch(self.ROC)
        for ep in self.ROC:
            Aggre = []
            for feat in self.ROC[ep]:
                plot = self.TemplateROC(self.OutDir + "ROC-Epoch", self.ROC[ep][feat]["FPR"], self.ROC[ep][feat]["TPR"])
                plot["Title"] = feat 
                Aggre.append(TLine(**plot))
            com = self.MergePlots(Aggre, self.OutDir + "ROC-Epoch")
            com.Title = "ROC Curve for Epoch " + str(ep)
            com.Filename = "Epoch_" + str(ep)
            com.SaveFigure() 
        
        for feat in self.AUC:
            self.SortEpoch(self.AUC[feat])
            plot = self.TemplateROC(self.OutDir, list(self.AUC[feat]), self.UnNestList(list(self.AUC[feat].values())))
            plot["Title"] = feat
            plot["xTitle"] = "Epoch"
            plot["yTitle"] = "Area Under ROC Curve"
            self.AUC[feat] = TLine(**plot)
        
        comb3 = self.MergePlots(list(self.AUC.values()), self.OutDir)
        comb3.Title = "Area under ROC Curve for All Features with respect to Epoch"
        comb3.Filename = "AUC_AllFeatures"
        comb3.SaveFigure()
        self.Plots["AUC"] = comb3
        
        self.MakeMassPlot(self.EdgeMass, "Edge", self.OutDir + "EdgeMass-Epoch")
        self.MakeMassPlot(self.NodeMass, "Node", self.OutDir + "NodeMass-Epoch")

        OutDir = self.OutDir + "ProcessReconstruction-Edge"
        self.Plots["EdgeProcessEfficiency"] = self.MakeReconstructionProcessEfficiency(self.EdgeMassPrcEff, OutDir)       
        self.Plots["EdgeEfficiency"] = self.MakeReconstructionEfficiency(self.EdgeMassAll, OutDir) 

        OutDir = self.OutDir + "ProcessReconstruction-Node"
        self.Plots["NodeProcessEfficiency"] = self.MakeReconstructionProcessEfficiency(self.NodeMassPrcEff, OutDir)      
        self.Plots["NodeEfficiency"] = self.MakeReconstructionEfficiency(self.NodeMassAll, OutDir)

        # ==== Write output log ===== #
        self._S = " | "
        
        def MakeLog(lines, outname, yTitle):
            for i in lines:
                i.yTitle = yTitle
            out = self.DumpTLines(lines)
            self.WriteText(out, self.OutDir + "/logs/" + outname)

        MakeLog(self.Plots["Accuracy"].Lines, "AccuracyOfAllFeatures", "%")
        MakeLog(self.Plots["Loss"].Lines, "LossOfAllFeatures", "a.u")
        MakeLog(self.Plots["AUC"].Lines, "AUC_AllFeatures", "a.u.c")
        
        lines = [] 
        lines += self.Plots["EdgeEfficiency"].Lines
        lines += self.Plots["NodeEfficiency"].Lines
        MakeLog(lines, "ReconstructionEfficiency", "%")
        
        for i in self.Plots["EdgeProcessEfficiency"]:
            MakeLog(self.Plots["EdgeProcessEfficiency"][i].Lines, "ProcessReconstructionEdgeFeature_"+i, "%")
        
        for i in self.Plots["NodeProcessEfficiency"]:
            MakeLog(self.Plots["NodeProcessEfficiency"][i].Lines, "ProcessReconstructionNodeFeature_"+i, "%")
 
        return self.Plots

class Train(Common):

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {} 
        self.VerboseLevel = 1
        self.Caller = "Train"


class Test(Common):

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {}
        self.VerboseLevel = 1
        self.Caller = "Test"

class All(Common): 

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {}
        self.VerboseLevel = 1
        self.Caller = "All"
