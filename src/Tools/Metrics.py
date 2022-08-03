from AnalysisTopGNN.IO import Directories, WriteDirectory, UnpickleObject
from AnalysisTopGNN.Plotting.Legacy import TGraph, CombineTGraph
import math
import matplotlib.pyplot as plt

class Metrics(Directories):

    def __init__(self, modelname, modelDir):
        self.Verbose = True
        Directories.__init__(self, modelDir)
        self.Caller = "METRICS"
        self.ModelName = modelname
        self.ModelDir = modelDir
        if self.ModelDir.endswith("/"):
            self.ModelDir = self.ModelDir[:len(self.ModelDir)-1]
        
        self.ModelStats = self.ModelDir + "/" + self.ModelName + "/Statistics"
        self.PlotDir = self.ModelDir + "/" + self.ModelName + "/Plots" 
        self.LogsDir = self.ModelDir + "/" + self.ModelName + "/Logs" 

        self.ReadStatisticDump(self.ModelStats)

        self.EpochTime = []
        self.kFoldTime_Nodes = {}
        self.kFold_n = {}
        
        self.TrainAcc = {}
        self.TrainLoss = {}
        self.ValidAcc = {}
        self.ValidLoss = {}

        self.AllTrainAcc = {}
        self.AllTrainLoss = {}
        self.AllValidAcc = {}
        self.AllValidLoss = {}

        self.Logs = []
        
        self.PlotFigures = True

    def ReadStatisticDump(self, outputDir = None):
        Files = self.ListFilesInDir(outputDir)
        self.Stats = {}
        self.Epochs = []
        for i in Files:
            epoch = i.split("_")[1].replace(".pkl", "")
            if epoch == "Done":
                self.File = UnpickleObject(i, outputDir)
                continue
            self.Stats[int(epoch)] = UnpickleObject(i, outputDir) 
            self.Epochs.append(int(epoch))
        self.Epochs.sort()
        self.Min = min(self.Epochs)
        self.Max = max(self.Epochs)

    def __CreateGraph(self, Title, xAxis, yAxis, xData, yData, name, errors = None, todir = None):
        
        L = TGraph()
        L.Color = "Blue"
        L.DefaultDPI = 250
        L.DefaultScaling = 10
        L.xTitle = xAxis
        L.yTitle = yAxis
        L.Title = Title
        L.xMax = max(xData)+0.5 
        L.xMin = min(xData)+0.5 
        L.xData = xData
        L.yData = yData
        L.Filename = name
        L.Init_PLT()

        if errors:
            L.ErrorBars = True
        
        if self.PlotFigures:
            if todir != None:
                L.Save(self.PlotDir + "/" + todir + "/")
            else:
                L.Save(self.PlotDir + "/")
        L.CloseFigure()
        return L

    def PlotStats(self, PlotDir = None):
        def FillLoop(inp, idx, key, node, kl, val):
            for TF in val[key]:
                if TF not in inp:
                    inp[TF] = {}
                if node not in inp[TF]:
                    inp[TF][node] = []
               
                inp[TF][node].append([])
                for y in val[key][TF][idx : idx + len(kl)]:
                    inp[TF][node][-1] += [float(p) for p in y]
            return inp

        def MergeFolds(inp, key, val, epc):
            for TF in val[key]:
                if TF not in inp:
                    inp[TF] = []
                if len(inp[TF]) < epc:
                    inp[TF].append([])

                for y in val[key][TF]:
                    inp[TF][-1] += [float(p) for p in y]
            return inp

        
        def DumpFigure(inpx, inpy, feature, Metric, Mode, Global = False, Log = False):
            Com = CombineTGraph()
            Com.Title = "Average " + Mode + " " + Metric + " for feature: " + feature
            Com.Filename = feature + "_" + Metric + "_" + Mode

            Com.Log = Log
            Com.DefaultDPI = 250
            Com.DefaultScaling = 10
            Com.LegendSize = 15
            Com.LabelSize = 15
            Com.FontSize = 10

            Com.ErrorBars = True
            col = ["b", "g", "r", "c", "m", "y"]
            it = 0
            for n in inpy:
                if Global == False:
                    L_ = self.__CreateGraph("Nodes-" + str(n), 
                         "Epoch", Metric, 
                         inpx, inpy[n], 
                         Mode + "_" + Metric + "_Node-" + str(n), 
                         errors = True, todir = feature + "/Raw")
                    L_.Title = "Nodes-" + str(n)
                else:
                    L_ = self.__CreateGraph(n, 
                         "Epoch", Metric, 
                         inpx, inpy[n], 
                         Mode + "_" + Metric + "_Feature_" + n, 
                         errors = True, todir = feature + "/Raw")
                    L_.Title = n
                try:
                    L_.Color = col[it]
                except:
                    L_.Color = it
                Com.Lines.append(L_)
                it += 1
            if self.PlotFigures:
                Com.Save(self.PlotDir + "/" + feature)

        def MergeLines(inpx, inpy1, inpy2, feature, Metric, Mode1, Mode2, Nodes = True, Log = False):
            Com = CombineTGraph()
            Com.Filename = Mode1 + "_" + Mode2 + "_" + Metric
            Com.Title = Mode1 + "(Dashed)/" + Mode2 + "(Solid) " + Metric + " " + feature
            Com.Log = Log
            Com.DefaultDPI = 250
            Com.DefaultScaling = 10
            Com.LegendSize = 15
            Com.LabelSize = 15
            Com.FontSize = 10
            self.PlotFigures = True
            it = 0

            col = ["b", "g", "r", "c", "m", "y"]
            for n in inpy1:
                if Nodes:
                    SubT = "Nodes-" + str(n)
                else:
                    SubT = str(n)
                L_1 = self.__CreateGraph(SubT,
                     "Epoch", Metric, 
                     inpx, inpy1[n], 
                     Mode1 + "_" + Metric + "_" + SubT, 
                     errors = True, todir = feature + "/Raw")
                try:
                    L_1.Color = col[it]
                except:
                    L_1.Color = it

                L_1.LineStyle = "dashed"

                L_2 = self.__CreateGraph(SubT, 
                     "Epoch", Metric, 
                     inpx, inpy2[n], 
                     Mode2 + "_" + Metric + "_" + SubT, 
                     errors = True, todir = feature + "/Raw")
                L_2.LineStyle = "solid"
                L_2.Marker = "x"
                try:
                    L_2.Color = col[it]
                except:
                    L_2.Color = it

                it+=1
                
                Com.Lines.append(L_1)
                Com.Lines.append(L_2)
            self.PlotFigures = True
            if self.PlotFigures:
                Com.Save(self.PlotDir + "/" + feature)


        if PlotDir != None:
            self.PlotDir = PlotDir
        
        for i in self.Epochs:
            vals = self.Stats[i]
            self.EpochTime.append(vals["EpochTime"][0])
            indx = 0

            for n in range(len(vals["Nodes"])):
                node = vals["Nodes"][n]
                FT = vals["FoldTime"][n]
                kF = vals["kFold"][n]

                 
                if node not in self.kFoldTime_Nodes:
                    self.kFoldTime_Nodes[node] = [[] for j in range(len(kF))]
                    self.kFold_n[node] = kF
                self.kFoldTime_Nodes[node] = [x + [y] for x, y in zip(self.kFoldTime_Nodes[node], FT)] 
                
                FillLoop(self.TrainAcc, indx, "Training_Accuracy", node, kF, vals)
                FillLoop(self.TrainLoss, indx, "Training_Loss", node, kF, vals)

                FillLoop(self.ValidAcc, indx, "Validation_Accuracy", node, kF, vals)
                FillLoop(self.ValidLoss, indx, "Validation_Loss", node, kF, vals)
                
                indx += len(kF) 
            
            MergeFolds(self.AllTrainAcc, "Training_Accuracy", vals, i)
            MergeFolds(self.AllTrainLoss, "Training_Loss", vals, i)

            MergeFolds(self.AllValidAcc, "Validation_Accuracy", vals, i)
            MergeFolds(self.AllValidLoss, "Validation_Loss", vals, i)

        # ======= Make Epoch Time plot ====== #
        self.__CreateGraph("Training + Validation Time", 
                "Epoch", "Time (s)", 
                self.Epochs, self.EpochTime, 
                "EpochTime", todir = "Time")

        # ======= Make kFold average time over all epochs ======= #
        for n in self.kFoldTime_Nodes:
            self.__CreateGraph("Average kFold Training and Validation Time for n-Nodes " + str(n), 
                    "k-Fold", "Average Time (s)", 
                    self.kFold_n[n], self.kFoldTime_Nodes[n], 
                    "kFoldTime_Node-" + str(n), 
                    errors = True, todir = "Time")
        
        # ======= Make Average Training/Validation Accuracy ======== #
        for ft in self.TrainAcc:
            DumpFigure(self.Epochs, self.TrainAcc[ft], ft, "Accuracy", "Training") 
            DumpFigure(self.Epochs, self.ValidAcc[ft], ft, "Accuracy", "Validation")
            MergeLines(self.Epochs, self.TrainAcc[ft], self.ValidAcc[ft], ft, "Accuracy", "Training", "Validation")
            
            DumpFigure(self.Epochs, self.TrainLoss[ft], ft, "Loss", "Training", Log = True)
            DumpFigure(self.Epochs, self.ValidLoss[ft], ft, "Loss", "Validation", Log = True)
            MergeLines(self.Epochs, self.TrainLoss[ft], self.ValidLoss[ft], ft, "Loss", "Training", "Validation", Log = True)   
        
        DumpFigure(self.Epochs, self.AllTrainAcc, "All-Features", "Accuracy", "Training", True) 
        DumpFigure(self.Epochs, self.AllValidAcc, "All-Features", "Accuracy", "Validation", True)
        MergeLines(self.Epochs, self.AllTrainAcc, self.AllValidAcc, "All-Features", "Accuracy", "Training", "Validation", False)
        
        DumpFigure(self.Epochs, self.AllTrainLoss, "All-Features", "Loss", "Training", True, Log = True)
        DumpFigure(self.Epochs, self.AllValidLoss, "All-Features", "Loss", "Validation", True, Log = True)
        MergeLines(self.Epochs, self.AllTrainLoss, self.AllValidLoss, "All-Features", "Loss", "Training", "Validation", False, True)         
        self.DumpLog()

    def DumpLog(self):
        def Record(string):
            self.Logs.append(string)

        def Average(inp):
            return float(sum(inp)/len(inp))

        def ToTime(inp):
            ori = inp
            h = int(inp/3600)
            inp = inp - h*3600
            m = int(inp/60)
            inp = inp - m*60
            s = int(inp)
            if h == m == s == 0:
                return str(round(ori, 3)) + "s"
            return str(h) + "h " + str(m) + "mins " + str(s) + "s"
        
        def RateOfChange(inp_l, feat, epch, Mode):
            av_p0 = Average(inp_l[feat][epch])
            if Mode == "Accuracy":
                av_p0 = av_p0*100

            if self.Max-1 == epch:
                return None, av_p0

            av_p1 = Average(inp_l[feat][epch+1])
            return (av_p1 - av_p0)/2, av_p0
            
            
        n_nodes = self.File["n_Node_Files"]
        N_n_nodes = self.File["n_Node_Count"]
        Tree_Dict = self.File["Tree"]
        Start_Dict = self.File["Start"]
        End_Dict = self.File["End"]
        ParticleLevel = self.File["Level"]
        SelfLoop = self.File["SelfLoop"]
        Samples = self.File["Samples"]
        
        Record("====== Sample Summary ======")
        All_Nodes = {}
        total = 0
        for i in range(len(n_nodes)):
            for n, k in zip(n_nodes[i], N_n_nodes[i]):
                if n not in All_Nodes:
                    All_Nodes[n] = 0
                All_Nodes[n] += k
                total += k

        Record("- Total Number of Events: " + str(total))
        for i in All_Nodes:
            Record("--> Nodes: " + str(i) + " (" + str(All_Nodes[i]) + ") " + str(round(float(All_Nodes[i]/total)*100, 2)) + "%")
        Record("- Names:") 
        for i in Samples:
            Record("--> " + i)
        Record("")

        Record("====== Sample Details ======")
        for i in range(len(Samples)):
            Record("__________________")
            Record("-> File Processed: " + Samples[i])
            Record("---> Tree: " + Tree_Dict[i] + " || Particle Level: " + ParticleLevel[i] + " || Graph Self Loop: " + str(SelfLoop[i]))
            Record("---> Start Index: " + str(Start_Dict[i]) + " || End Index: " + str(End_Dict[i]))  
            Record("---> n-Nodes: " + ", ".join([str(i) for i in n_nodes[i]]))
            Record("---> Number of n-Nodes: " + ", ".join([str(i) for i in N_n_nodes[i]]))
        
        self.WriteLog("SampleSummary")
     
        Record("======= Training Summary =======")
        Record("- Number of Epochs: " + str(self.Max - self.Min +1))
        Record("- Total Time Elapsed: " + ToTime(self.File["TrainingTime"]))
        Record("- Batch Size: " + str(self.File["BatchSize"]))
        Record("- Number of k-Folds:")
        for i in self.kFold_n: 
            Record(" (+)> Nodes: " + str(i) + " -> k = " + str(max(self.kFold_n[i])))
            Record("    > Average Fold Time: " + ToTime(Average([i for j in self.kFoldTime_Nodes for x in self.kFoldTime_Nodes[j] for i in x])))

        Record("- Feature Average Training Accuracy/Loss: ")
        for i in self.AllTrainAcc:
            av = [Average(k) for k in self.AllTrainAcc[i]]
            Record(" (+)> Feature: " + i)
            Record("    > Final: " + str(round(av[-1], 4)*100) + "%")
            Record("    > Best:  " + str(round(max(av), 4)*100) + 
                    "% at Epoch(s): " + ", ".join([ str(k+1) for k in range(len(av)) if av[k] == max(av) ]))
            av_l = [Average(k) for k in self.AllTrainLoss[i]]
            Record("    > Final: " + str(round(av_l[-1], 4)))
            Record("    > Best:  " + str(round(min(av_l), 4)) + 
                   " at Epoch(s): " + ", ".join([ str(k+1) for k in range(len(av_l)) if av_l[k] == min(av_l) ]))

        Record("- Feature Average Validation Accuracy/Loss: ")
        for i in self.AllValidAcc:
            av = [Average(k) for k in self.AllValidAcc[i]]
            Record(" (+)> Feature: " + i)
            Record("    > Final: " + str(round(av[-1], 4)*100) + "%")
            Record("    > Best:  " + str(round(max(av), 4)*100) + 
                    "% at Epoch(s): " + ", ".join([ str(k+1) for k in range(len(av)) if av[k] == max(av) ]))
            av_l = [Average(k) for k in self.AllValidLoss[i]]
            Record("    > Final: " + str(round(av_l[-1], 4)))
            Record("    > Best:  " + str(round(min(av_l), 4)) + 
                   " at Epoch(s): " + ", ".join([ str(k+1) for k in range(len(av_l)) if av_l[k] == min(av_l) ]))
        
        self.WriteLog("TrainingSummary")
        
        Record("======= Model Summary =======")
        Record("- Learning Rate: " + str(self.File["Model"]["LearningRate"]))
        Record("- Weight Decay: " + str(self.File["Model"]["WeightDecay"]))
        Record("- Model Function Name: " +  str(self.File["Model"]["ModelFunctionName"]).split("'")[1])

        self.WriteLog("ModelSummary")
        
        Record("======= Detailed Training Log =======")
        Features = [i for i in self.AllTrainAcc]
        Record("---- (Training/Validation) Feature :: Accuracy :: Loss --- ")

        Delta_T = {}
        Delta_V = {}
        Delta_T_V = {}
        for i in Features:
            Delta_T[i + "::Accuracy"] = []
            Delta_T[i + "::Loss"] = []

            Delta_V[i + "::Accuracy"] = []
            Delta_V[i + "::Loss"] = []

            Delta_T_V[i + "::Accuracy"] = []
            Delta_T_V[i + "::Loss"] = []


        for i in range(self.Min-1, self.Max):
            Record(" -> EPOCH = " + str(i+1)) 
            for ft in Features:
                r_ta, av_ta = RateOfChange(self.AllTrainAcc, ft, i, "Accuracy")
                r_tl, av_tl = RateOfChange(self.AllTrainLoss, ft, i, "Loss")
                r_va, av_va = RateOfChange(self.AllValidAcc, ft, i, "Accuracy")
                r_vl, av_vl = RateOfChange(self.AllValidLoss, ft, i, "Loss")
                
                if r_ta != None:
                    Delta_T[ft + "::Accuracy"].append(r_ta)
                    Delta_T[ft + "::Loss"].append(r_tl)

                    Delta_V[ft + "::Accuracy"].append(r_va)
                    Delta_V[ft + "::Loss"].append(r_vl)
                    
                    try: 
                        Delta_T_V[ft + "::Accuracy"].append(r_ta / r_va)
                    except ZeroDivisionError:
                        Delta_T_V[ft + "::Accuracy"].append(r_ta / (r_va +1))

                    try: 
                        Delta_T_V[ft + "::Loss"].append(r_tl / r_vl)
                    except ZeroDivisionError:
                        Delta_T_V[ft + "::Loss"].append(r_tl / (r_vl + 1))





                Record("      " + ft + " :: " + str(round(av_ta, 3)) + "% :: " + str(round(av_tl, 3)) + " / " + str(round(av_va, 3)) + "% :: " + str(round(av_vl, 3)))

        Record("")
        Record("---- Average (Training::Validation) Accuracy Improvement Per Epoch ----")
        for i in Features:
            Record("-> " + i + ": " + str(round(float(sum(Delta_T[i+"::Accuracy"])/len(Delta_T[i+"::Accuracy"])), 4)) + 
                             " :: " + str(round(float(sum(Delta_V[i+"::Accuracy"])/len(Delta_V[i+"::Accuracy"])), 4)))
        
        Record("")
        Record("---- Average (Training::Validation) Loss Improvement Per Epoch ----")
        for i in Features:
            Record("-> " + i + ": " + str(round(float(sum(Delta_T[i+"::Loss"])/len(Delta_T[i+"::Loss"])), 4)) + 
                             " :: " + str(round(float(sum(Delta_V[i+"::Loss"])/len(Delta_V[i+"::Loss"])), 4)))

        Record("")
        Record("---- Average dT/dV Accuracy Improvement ----")
        for i in Features:
            Record("-> " + i + ": " + str(round(float(sum(Delta_T_V[i+"::Accuracy"])/len(Delta_T_V[i+"::Accuracy"])), 4)))

        Record("")
        Record("---- Average dT/dV Loss Improvement ----")
        for i in Features:
            Record("-> " + i + ": " + str(round(float(sum(Delta_T_V[i+"::Loss"])/len(Delta_T_V[i+"::Loss"])), 4)))


        self.WriteLog("TrainingDetails")


        Com_A = CombineTGraph()
        Com_A.Title = "Accuracy dT/dV plot"
        Com_L = CombineTGraph()
        Com_L.Title = "Loss dT/dV plot"
        for ft in Features:
            L_ = self.__CreateGraph(ft, 
                    "Epoch", "dT/dV", 
                    [i+1 for i in range(self.Min-1, self.Max-1)], Delta_T_V[ft + "::Accuracy"], 
                    ft + "_A", todir = "dT_dV/Raw")
            Com_A.Lines.append(L_) 

            L_ = self.__CreateGraph(ft, 
                    "Epoch", "dT/dV", 
                    [i+1 for i in range(self.Min-1, self.Max-1)], Delta_T_V[ft + "::Loss"], 
                    ft + "_L", todir = "dT_dV/Raw")
            Com_L.Lines.append(L_) 
       
        Com_A.Filename = "Accuracy_dT_dV"
        Com_L.Filename = "Loss_dT_dV"
        Com_A.Save(self.PlotDir + "/dT_dV")
        Com_L.Save(self.PlotDir + "/dT_dV")

    
    def WriteLog(self, name):
        opt = ""
        for i in self.Logs:
            opt += i + "\n"
        WriteDirectory().WriteTextFile(opt, self.LogsDir, name)
        self.Logs = []

