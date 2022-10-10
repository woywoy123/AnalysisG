from AnalysisTopGNN.Generators import EventGenerator
from AnalysisTopGNN.Plotting.TemplateHistograms import TH1FStack
from AnalysisTopGNN.Plotting.TemplateLines import TLineStack, TLine
from AnalysisTopGNN.IO import WriteDirectory
from collections import Counter

class GraphicsCompiler:

    def __init__(self):
        self.MakeSampleNodesPlot = True
        self.MakeTrainingPlots = True
        self.MakeStaticHistograms = False
        self.pwd = ""
    
    def SampleNodes(self, dict_stat):
       
        Plots = {
                    "Title" : "Superimposed Graph Nodes Distribution for Training/Test Samples", 
                    "Filename" : "n_Nodes_All", 
                    "Normalize" : "%",
                    "yTitle" : "Percentage of n-Nodes (%)", 
                    "xTitle" : "Number of Nodes", 
                    "Histogram" : "All", 
                    "Histograms" : ["Training", "Test"], 
                    "xBinCentering" : True, 
                    "Style" : "ATLAS",
                    "Data" : dict_stat
                }
        
        if self.MakeSampleNodesPlot == False:
            return 
        
        x = TH1FStack(**Plots)
        x.SaveFigure(self.pwd + "/NodeStatistics")
        
        Proc_All, Proc_Tr, Proc_Te = {"All" : []}, {"All" : []}, {"All" : []}
        for p_ in list(dict_stat):
            al = [node for node in dict_stat[p_]["All"] for qnt in range(dict_stat[p_]["All"][node])]
            tr = [node for node in dict_stat[p_]["Training"] for qnt in range(dict_stat[p_]["Training"][node])]
            te = [node for node in dict_stat[p_]["Test"] for qnt in range(dict_stat[p_]["Test"][node])]

            Proc_All |= {p_ : al}
            Proc_Tr |= {p_ : tr}
            Proc_Te |= {p_ : te}
       
            Proc_All["All"] += al
            Proc_Tr["All"] += tr
            Proc_Te["All"] += te
 
        Plots["Filename"] = "n_Nodes_All_Processes"
        Plots["Title"] = "Superimposed Graph Nodes Distribution \nfor Simulated Processes - Complete Sample"
        Plots["Histograms"] = list(dict_stat)
        Plots["Data"] = Proc_All
        x = TH1FStack(**Plots)
        x.SaveFigure(self.pwd + "/NodeStatistics")

        Plots["Filename"] = "n_Nodes_Training_Processes"
        Plots["Title"] = "Superimposed Graph Nodes Distribution \nfor Simulated Processes - Training Sample"
        Plots["Histograms"] = list(dict_stat)
        Plots["Data"] = Proc_Tr
        x = TH1FStack(**Plots)
        x.SaveFigure(self.pwd + "/NodeStatistics")

        Plots["Filename"] = "n_Nodes_Test_Processes"
        Plots["Title"] = "Superimposed Graph Nodes Distribution \nfor Simulated Processes - Test Sample"
        Plots["Histograms"] = list(dict_stat)
        Plots["Data"] = Proc_Te
        x = TH1FStack(**Plots)
        x.SaveFigure(self.pwd + "/NodeStatistics")

    def TrainingPlots(self, model_dict, model):
        if self.MakeTrainingPlots == False:
            return 

        Plots = {
                    "Title" : "Epoch Time", 
                    "Filename" : "EpochTime", 
                    "yTitle" : "Time (s)", 
                    "xTitle" : "Epoch", 
                    "Style" : "ATLAS",
                    "Lines" : ["EpochTime"],
                    "yData" : ["EpochTime"],
                    "xData" : ["Epochs"],
                    "Data" : model_dict[model], 
                } 
       

        x = TLineStack(**Plots) 
        x.SaveFigure(self.pwd)
        
        Plots["Title"] = "Average K-Fold Time"
        Plots["Filename"] = "kFold_Time"
        Plots["xTitle"] = "k-Fold"
        Plots["Lines"] = ["k-Fold"]
        Plots["yData"] = ["kFoldTime"]
        Plots["xData"] = ["kFold"]
        Plots["DoStatistics"] = True
        Plots["OutputDirectory"] = self.pwd
        Plots["MakeStaticHistograms"] = self.MakeStaticHistograms
        x = TLineStack(**Plots)
        x.SaveFigure(self.pwd)
            
        for out in model_dict[model]["Outputs"]:
            Plots["xTitle"] = "Epoch"
            Plots["DoStatistics"] = True
            Plots["Data"] = model_dict[model][out]
            Plots["Lines"] = ["Training", "Validation"]
            
            Plots["Title"] = "Accuracy of feature: " + out
            Plots["Filename"] = "Accuracy_" + out
            Plots["yTitle"] = "Accuracy of Prediction"
            Plots["yData"] = ["TrainingAccuracy", "ValidationAccuracy"]
            Plots["xData"] = ["TrainingEpochs", "ValidationEpochs"]
            x = TLineStack(**Plots)
            x.SaveFigure(self.pwd)
 
            Plots["Title"] = "Loss of feature: " + out
            Plots["Filename"] = "Loss_" + out
            Plots["yTitle"] = "Loss"
            Plots["yData"] = ["TrainingLoss", "ValidationLoss"]
            Plots["xData"] = ["TrainingEpochs", "ValidationEpochs"]
            x = TLineStack(**Plots)
            x.SaveFigure(self.pwd)
    
    def TestPlots(self, stat_dict, model):
        for out in stat_dict[model]["Outputs"]:
            Plots = {} 
            Plots["xTitle"] = "Epoch"
            Plots["Lines"] = ["Test"]
            Plots["xData"] = ["TestEpochs"]
            Plots["DoStatistics"] = True
            Plots["Data"] = stat_dict[model][out]
            Plots["OutputDirectory"] = self.pwd

            Plots["Title"] = "Accuracy of feature: " + out + " on Test (Withheld Data)"
            Plots["Filename"] = "Test_Accuracy_" + out
            Plots["yTitle"] = "Accuracy of Prediction"
            Plots["yData"] = ["TestAccuracy"]
            x = TLineStack(**Plots)
            x.SaveFigure()
 
            Plots["Title"] = "Loss of feature: " + out + " on Test (Withheld Data)"
            Plots["Filename"] = "Test_Loss_" + out
            Plots["yTitle"] = "Loss"
            Plots["yData"] = ["TestLoss"]
            x = TLineStack(**Plots)
            x.SaveFigure()

    def ROCCurve(self, ROC_val):
        for feature in ROC_val:
            for epoch in ROC_val[feature]:
                models = list(ROC_val[feature][epoch])
                Plots = {}
                Plots["xTitle"] = "False Positive Rate"
                Plots["yTitle"] = "True Positive Rate"
                Plots["Title"] = "Receiver Operating Characteristic (ROC) Curve at Epoch: " + str(epoch) + " for feature: " + feature
                Plots["Lines"] = [m + " - AUC " + str(round(ROC_val[feature][epoch][m]["auc"], 3)) for m in models]
                Plots["xData"] = [i + "/fpr" for i in models]
                Plots["yData"] = [i + "/tpr" for i in models]
                Plots["Filename"] = "ROC_" + str(epoch)
                Plots["Data"] = ROC_val[feature][epoch]
                Plots["ROC"] = True
                x = TLineStack(**Plots)
                x.xMin = 0
                x.yMin = 0
                x.xMax = 1
                x.yMax = 1
                x.SaveFigure(self.pwd + "/ROC_" + feature)
    
    def ParticleReconstruction(self, mass_dict):

        for feat in mass_dict:
            if feat == "Process":
                continue
            
            PerfProcess = {}
            Epochs =  [ep for ep in mass_dict[feat] if ep != "Truth"]
            Epochs.sort()

            for epoch in Epochs:
                for model in mass_dict[feat][epoch]:
                    dic = mass_dict[feat][epoch][model]
                    
                    Plot = {"xTitle" : "Invariant Mass (GeV)", "yTitle" : "Entries", "xBins" : 500, "xMin" : 0, "xMax" : 250}
                    Plot["Title"] = "Reconstructed Invariant Top Mass from Feature: " + feat + " at Epoch " + str(epoch)
                    Plot["Histograms"] = [ {"Title" : model, "xData" : dic["Tops"]} ]
                    Plot["OutputDirectory"] = self.pwd + "/TopMass/" + feat + "/" + model + "/"
                    Plot["Filename"] = "EPOCH_" + str(epoch)
                    
                    if "Truth" in mass_dict[feat]:
                        Plot["Histogram"] =  {"Title" : "Truth", "xData" : mass_dict[feat]["Truth"]["Tops"]} 
                    x = TH1FStack(**Plot)
                    x.SaveFigure()
                    
                    if "Truth" not in mass_dict[feat]:
                        continue
                    Plot = {} 
                    Plot["Title"] = "Event Reconstruction Efficiency: " + feat + " at Epoch " + str(epoch)
                    Plot["xTitle"] = "Efficiency (%)"
                    Plot["yTitle"] = "Entries"
                    Plot["OutputDirectory"] = self.pwd + "/EventEfficiency/" + feat + "/" + model + "/"
                    Plot["Filename"] = "EPOCH_" + str(epoch)
                    Plot["Histograms"] = [ {"Title" : model, "xData" : dic["EventEfficiency"]}]
                    Plot["xMin"] = 0
                    Plot["xMax"] = 150
                    Plot["xBins"] = 150
                    x = TH1FStack(**Plot)
                    x.SaveFigure()

                    proc = { i : float(k/j)*100 for i, k, j in zip(dic["ProcessPredTops"], dic["ProcessPredTops"].values(), dic["ProcessTruthTops"].values()) }
                    Plot = {} 
                    Plot["Title"] = "Top Reconstruction Efficiency by Process: " + feat + " at Epoch " + str(epoch)
                    Plot["xTitle"] = "Process"
                    Plot["yTitle"] = "Efficiency (%)"
                    Plot["OutputDirectory"] = self.pwd + "/ProcessEfficiency/" + feat + "/" + model + "/"
                    Plot["Filename"] = "EPOCH_" + str(epoch)
                    Plot["yData"] = list(proc.values())
                    Plot["xData"] = [i for i in range(len(proc))]
                    Plot["xTickLabels"] = list(proc)
                    Plot["yMax"] = 110
                    Plot["yMin"] = 0
                    x = TLine(**Plot)
                    x.SaveFigure()
                    
                    if model not in PerfProcess:
                        PerfProcess[model] = {}
                        PerfProcess[model]["Epoch"] = []
                        PerfProcess[model]["All_Epochs"] = []
                        PerfProcess[model]["Event_Eff"] = []
                    for pr in proc:
                        if pr not in PerfProcess[model]:
                            PerfProcess[model][pr] = []
                        PerfProcess[model][pr] += [proc[pr]]
                    PerfProcess[model]["Epoch"] += [epoch]
                    PerfProcess[model]["All_Epochs"] += [epoch]*len(dic["EventEfficiency"])
                    PerfProcess[model]["Event_Eff"] += dic["EventEfficiency"]
             
            for model in PerfProcess:
                Plot = {} 
                Plot["Title"] = "Top Reconstruction Efficiency by Process for " + model + " and feature: " + feat
                Plot["xTitle"] = "Epoch"
                Plot["yTitle"] = "Efficiency (%)"
                Plot["OutputDirectory"] = self.pwd + "/Process/" + feat + "/"
                Plot["Filename"] = model
                Plot["Lines"] = [i for i in PerfProcess[model] if i != "Epoch"]
                Plot["xData"] = ["Epoch"]*(len(list(PerfProcess[model]))-1)
                Plot["yData"] = [i for i in PerfProcess[model] if i != "Epoch"]
                Plot["yMax"] = 101
                Plot["yMin"] = -1
                Plot["Data"] = PerfProcess[model]
                x = TLineStack(**Plot)
                x.SaveFigure()
            
            Plot["Title"] = "Overall Top Reconstruction Efficiency of Models for Feature: " + feat
            Plot["OutputDirectory"] = self.pwd + "/ModelComparison/" 
            Plot["Filename"] = feat
            Plot["Lines"] = list(PerfProcess)
            Plot["xData"] = [m + "/All_Epochs" for m in PerfProcess]
            Plot["yData"] = [m + "/Event_Eff" for m in PerfProcess]
            Plot["Data"] = PerfProcess
            Plot["DoStatistics"] = True
            Plot["MakeStaticHistograms"] = False
            y = TLineStack(**Plot)
            y.SaveFigure()


class LogCompiler(WriteDirectory):

    def __init__(self):
        self._H = "="*5
        self._B = "-->"
        self._S = " | "

    def __MakeTable(self, inpt):
        
        def FindMaxCols(col):
            range_col = [0 for i in col[0]]
            for i in range(len(col)):
                for j in range(len(col[i])):
                    cur = range_col[j] 
                    if len(col[i][j]) > cur:
                        range_col[j] = len(col[i][j])
            return range_col

        def OptimizeSegment(seg):
            
            Bullets = []
            cols = []
            for i in seg:
                tmp = i.split(self._S)[0].split(":")
                Bullets.append([tmp[0].split(self._B)[-1]])
                t = [tmp[1]]
                t += [t for t in i.split(self._S)[1:]]
                cols.append(t)
            col = FindMaxCols(cols)
            bull = FindMaxCols(Bullets)
            
            out = []
            for i in range(len(Bullets)):
                B = " "*int(bull[0] - len(Bullets[i][0]))
                line = self._B + " " + B + Bullets[i][0] + ": "
                line += self._S.join([" "*int(col[k] - len(cols[i][k])) + cols[i][k] for k in range(len(col))])
                out.append(line)
            return out

        segment = {}
        segment_H = {}
        it = 0
        for i in inpt:
            if i.count(self._H) == 2:
                it += 1
                segment[it] = []
                segment_H[it] = i
                continue
            segment[it].append(i)
       
        out = []
        for i in segment:
            out += [segment_H[i]]
            if sum([k.count(self._S) for k in segment[i]]) > 0: 
                out += OptimizeSegment(segment[i])
                continue
            out += segment[i]
            out += [""]
        return out
        
    def Heading(self, Text):
        return self._H + " " + Text + " " + self._H

    def Percent(self, lst1, lst2):
        if isinstance(lst1, list): 
            val1, val2 = len(lst1), len(lst2)
        else:
            val1, val2 = lst1, lst2
        return "(" + str(round(float(val1/val2)*100, 3)) +"%" +")"
    
    def Bullet(self, text):
        return self._B + " " + text + ": "
    
    def Len(self, lst):
        return str(len(lst))

    def __Recursive(self, inpt, search):
        if isinstance(inpt, dict) == False:
            return inpt
        if search in inpt:
            out = []
            for k in inpt[search]:
                if isinstance(inpt[search][k], list):
                    out += inpt[search][k]
                else:
                    out += [k]*inpt[search][k]
            return out
        return [l for i in inpt for l in self.__Recursive(inpt[i], search)]

    def SampleNodes(self, dict_stat):
        All = self.__Recursive(dict_stat, "All")
        Tr = self.__Recursive(dict_stat, "Training") 
        Te = self.__Recursive(dict_stat, "Test")

        Proc_All, Proc_Tr, Proc_Te = {}, {}, {}
        for p_ in list(dict_stat):
            Proc_All |= {p_ : [node for node in dict_stat[p_]["All"] for qnt in range(dict_stat[p_]["All"][node])]}
            Proc_Tr |= {p_ : [node for node in dict_stat[p_]["Training"] for qnt in range(dict_stat[p_]["Training"][node])]}
            Proc_Te |= {p_ : [node for node in dict_stat[p_]["Test"] for qnt in range(dict_stat[p_]["Test"][node])]}
        
        text = []
        text += [self.Heading("Sample Splitting")]
        text += [self.Bullet("Sample Size") + self.Len(All) + self._S]
        text += [self.Bullet("Training Size") + self.Len(Tr) + self._S]
        text += [self.Bullet("Test Size") + self.Len(Te) + self._S]

        text += [self.Heading("Sample Composition (All | Training | Test)")]
        for i in Proc_All:
            col = self.Bullet(i)
            col += self.Len(Proc_All[i]) + " " + self.Percent(Proc_All[i], All) + self._S
            col += self.Len(Proc_Tr[i]) + " " + self.Percent(Proc_Tr[i], Tr) + self._S
            col += self.Len(Proc_Te[i]) + " " + self.Percent(Proc_Te[i], Te) + self._S
            text.append(col)

        text += [self.Heading("Sample Node Composition")]

        nodes = list(Counter(All))
        nodes.sort()
        nodes = [str(i) for i in nodes] 
        text += [self.Bullet("n-Nodes") + self._S.join(nodes) + self._S]
        for i in Proc_All:
            col = self.Bullet(i)

            _all = dict(Counter(All))
            nod = [_all[int(k)] if int(k) in _all else 0 for k in nodes]
             
            p_all = dict(Counter(Proc_All[i]))
            per = self._S.join([self.Percent(p_all[int(k)], _all[int(k)]) if int(k) in p_all else "(" + str(0) + "%)" for k in nodes]) + self._S
            
            col += per
            text.append(col)
        self.WriteTextFile(self.__MakeTable(text), "NodeStatistics", "SampleNodesDetails")
