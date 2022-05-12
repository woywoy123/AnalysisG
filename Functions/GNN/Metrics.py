from Functions.IO.Files import Directories, WriteDirectory
from Functions.Plotting.Histograms import TGraph, CombineTGraph
from Functions.IO.IO import UnpickleObject
import math

class Metrics(Directories):

    def __init__(self, modelname, modelDir):
        self.Verbose = True
        Directories.__init__(self, modelDir)
        self.Caller = "METRICS"
        self.ModelName = modelname
        self.ModelDir = modelDir
        if self.ModelDir.endswith("/"):
            self.ModelStats = self.ModelDir + self.ModelName + "/Statistics"
            self.PlotDir = self.ModelDir + self.ModelName + "/Plots"
        else:
            self.ModelStats = self.ModelDir + "/" + self.ModelName + "/Statistics"
            self.PlotDir = self.ModelDir + "/" + self.ModelName + "/Plots" 

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
        
        self.PlotFigures = False

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

    def __CreateGraph(self, Title, xAxis, yAxis, xData, yData, name, errors = None, node = None):
        
        L = TGraph()
        L.Color = "Blue" 
        L.xTitle = xAxis
        L.yTitle = yAxis
        L.Title = Title
        L.xMax = self.Max+0.5
        L.xMin = self.Min+0.5
        L.xData = xData
        L.yData = yData
        L.Filename = name
        if self.PlotFigures == False:
            return L
        if errors:
            L.ErrorBars = True
        if node != None:
            L.Save(self.PlotDir + "/Raw" + "/Node-" + str(node))
        else:
            L.Save(self.PlotDir + "/Raw")
        return L

    def PlotStats(self, PlotDir = None):
        def FillLoop(inp, idx, key, node, kl, val):
            for TF in val[key]:
                if TF not in inp:
                    inp[TF] = {}
                if node not in inp[TF]:
                    inp[TF][node] = [[] for j in range(len(kl))]
                inp[TF][node] = [x + y for x, y in zip(inp[TF][node], val[key][TF][idx : idx + len(kl)])]
            return inp

        def MergeFolds(inp, key, val, epc):
            for TF in val[key]:
                if TF not in inp:
                    inp[TF] = [[]]
                if len(inp[TF]) < epc:
                    inp[TF].append([])
                inp[TF][-1] += [x[0] for x in val[key][TF]]
            return inp

        
        def DumpFigure(inpx, inpy, feature, Metric, Mode, Global = False):
            Com = CombineTGraph()
            Com.Title = Mode + " " + Metric
            Com.Log = False
            for n in inpy:
                if Global == False:
                    L_ = self.__CreateGraph("Average " + Mode + " " + Metric + " for n-Nodes " + str(n) + " for feature: " + feature, 
                         "Epoch", Metric, inpx, inpy[n], 
                         Mode + "_" + Metric + "_Node-" + str(n) + "_Feature_" + feature, True, n)
                    L_.Title = "Nodes-" + str(n)
                else:
                    L_ = self.__CreateGraph("Average " + Mode + " " + Metric + " for feature: " + n, 
                         "Epoch", Metric, inpx, inpy[n], 
                         Mode + "_" + Metric + "_Feature_" + n, True)
                    L_.Title = n

                Com.Lines.append(L_)
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
                

            MergeFolds(self.AllTrainAcc, "Training_Accuracy", vals, i)
            MergeFolds(self.AllTrainLoss, "Training_Loss", vals, i)

            MergeFolds(self.AllValidAcc, "Validation_Accuracy", vals, i)
            MergeFolds(self.AllValidLoss, "Validation_Loss", vals, i)
            indx += len(kF) 

        # ======= Make Epoch Time plot ====== #
        self.__CreateGraph("Training + Validation Time", "Epoch", "Time (s)", self.Epochs, self.EpochTime, "EpochTime")

        # ======= Make kFold average time over all epochs ======= #
        for n in self.kFoldTime_Nodes:
            self.__CreateGraph("Average kFold Training and Validation Time for n-Nodes " + str(n), 
                    "kFold", "Average Time (s)", self.kFold_n[n], self.kFoldTime_Nodes[n], 
                    "kFoldTime_Node-" + str(n), True, n)
        
        # ======= Make Average Training/Validation Accuracy ======== #
        for ft in self.TrainAcc:
            DumpFigure(self.Epochs, self.TrainAcc[ft], ft, "Accuracy", "Training") 
            DumpFigure(self.Epochs, self.ValidAcc[ft], ft, "Accuracy", "Validation")
            DumpFigure(self.Epochs, self.TrainLoss[ft], ft, "Loss", "Training")
            DumpFigure(self.Epochs, self.ValidLoss[ft], ft, "Loss", "Validation")
        
        DumpFigure(self.Epochs, self.AllTrainAcc, "AllFeatures", "Accuracy", "Training") 
        DumpFigure(self.Epochs, self.AllValidAcc, "AllFeatures", "Accuracy", "Validation")
        DumpFigure(self.Epochs, self.AllTrainLoss, "AllFeatures", "Loss", "Training")
        DumpFigure(self.Epochs, self.AllValidLoss, "AllFeatures", "Loss", "Validation")
        
        self.DumpLog()

    def DumpLog(self):
        def Average(inp):
            return float(sum(inp)/len(inp))
        average_epoch_time = str(round(Average(self.EpochTime), 2))
        Tree = "nominal"
        ParticleLevel = "xxx"
        FileIndex = ""
        n_nodes = self.Stats[self.Min]["Nodes"]
       
        print(self.File)




        return 
        





        pass
    
        batch_stat = str(round(sum(self.Sample.BatchTime) / len(self.Sample.BatchTime), 2))
        
        Output = "===== Sample Information ===== \n"
        for i in self.Sample.Loader.Bundles:
            Output += "-> Tree Processed: " + i[0] + "\n"
            Output += "-> Mode: " + i[-1] + "\n"
            Bundle = i[1].FileEventIndex
            for j in Bundle:
                Output += "+-> File Processed: " + j + "\n"
                Output += "+-> Start: " + str(Bundle[j][0]) + " End: " + str(Bundle[j][1]) + "\n"
            
        total = 0
        Output += "\nNumber of Nodes\n"
        for nodes in self.Sample.DataLoader:
            data = self.Sample.DataLoader[nodes]
            Output += "-> Nodes: " + str(nodes) + " - Size: " + str(len(data))+ "\n"
            total += len(data)
        Output += "Total Sample Size: " + str(total) + "\n\n"

        Output += "===== Training Information ===== " + "\n"
        Output += "Number of Epochs: " + str(self.Sample.Epochs) + "\n"
        Output += "Time taken for all Epochs: " + str(round(self.Sample.TrainingTime, 2)) + "s" + "\n"
        Output += "Average Epoch Time: " + batch_stat + "s \n"
        Output += "Longest Epoch Time: " + str(round(max(self.Sample.BatchTime), 2)) + "s" + "\n"
        Output += "K-Fold: " + str(self.Sample.kFold) + "\n"
        Output += "Batch Size: " + str(self.Sample.DefaultBatchSize) + "\n"
        Output += "Device: " + str(self.Sample.Device_s) + "\n"
        Output += "Final Training Accuracy: " + str(round(self.__Stats(self.__TrainStatistics[self.Sample.Epochs])*100, 4)) + "%" + "\n"
        Output += "Final Training Loss: " + str(round(self.__Stats(self.__LossTrainStatistics[self.Sample.Epochs][-1]), 4)) + "" + "\n"
        Output += "Final Validation Accuracy: " + str(round(self.__Stats(self.__ValidationStatistics[self.Sample.Epochs])*100, 4)) + "%" + "\n"
        Output += "Final Validation Loss: " + str(round(self.__Stats(self.__LossValidationStatistics[self.Sample.Epochs][-1]), 4)) + "" + "\n\n"
        
        Output += "===== Model Information ===== \n"
        Output += "Learning Rate: " + str(self.Sample.LearningRate) + "\n"
        Output += "Weight Decay: " + str(self.Sample.WeightDecay) + "\n"
        Output += "Target Type: " + self.Sample.DefaultTargetType + "\n"
        Output += "Loss Function: " + self.Sample.DefaultLossFunction + "\n"
        Output += "Model Name: " + self.Sample.TrainingName + "\n\n"

        def GenericText(_Last, _First, Str1, Str2):
            Val = self.__Stats(_Last, _First)
            out = ""
            if Val < 0:
                out += Str1 + " Decrease: " + str(round(abs(Val), 4)) + Str2 + " "
            else:                                                       
                out += Str1 + " Increase: " + str(round(abs(Val), 4)) + Str2 + " "
            return out

        Output += "====== Loss Performance From Epoch 1 -> " + str(self.Sample.Epochs) + "====== \n "
        Delta_LV, Delta_LT = [], []
        for i in range(self.Sample.Epochs-1):
            LV_Last, LV_First = self.__LossValidationStatistics[i+1], self.__LossValidationStatistics[i+2]
            LT_Last, LT_First = self.__LossTrainStatistics[i+1], self.__LossTrainStatistics[i+2]

            Output += "[" + str(i+1) + "/" + str(self.Sample.Epochs) + "]  " + GenericText(LV_Last, LV_First, "Validation Loss", "%") + "|| " + GenericText(LT_Last, LT_First, "Training Loss", "%\n")
            
            Delta_LV.append(self.__Stats(LV_Last, LV_First)) 
            Delta_LT.append(self.__Stats(LT_Last, LT_First))
       

        Output += GenericText(Delta_LV, "", "---- Average Validation Loss", "% / Epoch \n")
        Output += GenericText(Delta_LT, "", "---- Average Training Loss", "% / Epoch \n")

        Output += "\n====== Accuracy Performance From Epoch 1 -> " + str(self.Sample.Epochs) + " ====== \n "
        Delta_LV, Delta_LT = [], []
        for i in range(self.Sample.Epochs-1):
            AV_Last, AV_First = self.__ValidationStatistics[i+1], self.__ValidationStatistics[i+2]
            AT_Last, AT_First = self.__TrainStatistics[i+1], self.__TrainStatistics[i+2]

            Output += "[" + str(i+1) + "/" + str(self.Sample.Epochs) + "] " + GenericText(AV_Last, AV_First, "Validation Accuracy", "%") + "|| " + GenericText(AT_Last, AT_First, "Training Accuracy", "%\n")

            Delta_LV.append(self.__Stats(AV_Last, AV_First)) 
            Delta_LT.append(self.__Stats(AT_Last, AT_First))

        Output += GenericText(Delta_LV, "", "---- Average Validation Accuracy", "% / Epoch \n")
        Output += GenericText(Delta_LT, "", "---- Average Training Accuracy", "% / Epoch \n")
       
        WriteDirectory().WriteTextFile(Output, Dir, "ModelInformation.txt")
































#from Functions.GNN.Graphs import GenerateDataLoader
#from Functions.Plotting.Histograms import TH1F, TH2F, TGraph, CombineTGraph
#from Functions.Tools.Alerting import Notification
#from Functions.GNN.Optimizer import Optimizer
#from Functions.IO.Files import WriteDirectory

#class EvaluationMetrics(Notification):
#
#    def __init__(self):
#        Notification.__init__(self)
#        self.Sample = []
#        self.__TruthAttribute = []
#        self.__PredictionAttribute = []
#        self.__Type = ""
#        self.__LossStatistics = ""
#        self.__Pairs = {}
#        self.__Truth = []
#        self.__Pred = []
#        self.Caller = "::EvaluationMetrics"
#        self.__Processed = False
#
#    def Receipt(self, Dir):
#        batch_stat = str(round(sum(self.Sample.BatchTime) / len(self.Sample.BatchTime), 2))
#        
#        Output = "===== Sample Information ===== \n"
#        for i in self.Sample.Loader.Bundles:
#            Output += "-> Tree Processed: " + i[0] + "\n"
#            Output += "-> Mode: " + i[-1] + "\n"
#            Bundle = i[1].FileEventIndex
#            for j in Bundle:
#                Output += "+-> File Processed: " + j + "\n"
#                Output += "+-> Start: " + str(Bundle[j][0]) + " End: " + str(Bundle[j][1]) + "\n"
#            
#        total = 0
#        Output += "\nNumber of Nodes\n"
#        for nodes in self.Sample.DataLoader:
#            data = self.Sample.DataLoader[nodes]
#            Output += "-> Nodes: " + str(nodes) + " - Size: " + str(len(data))+ "\n"
#            total += len(data)
#        Output += "Total Sample Size: " + str(total) + "\n\n"
#
#        Output += "===== Training Information ===== " + "\n"
#        Output += "Number of Epochs: " + str(self.Sample.Epochs) + "\n"
#        Output += "Time taken for all Epochs: " + str(round(self.Sample.TrainingTime, 2)) + "s" + "\n"
#        Output += "Average Epoch Time: " + batch_stat + "s \n"
#        Output += "Longest Epoch Time: " + str(round(max(self.Sample.BatchTime), 2)) + "s" + "\n"
#        Output += "K-Fold: " + str(self.Sample.kFold) + "\n"
#        Output += "Batch Size: " + str(self.Sample.DefaultBatchSize) + "\n"
#        Output += "Device: " + str(self.Sample.Device_s) + "\n"
#        Output += "Final Training Accuracy: " + str(round(self.__Stats(self.__TrainStatistics[self.Sample.Epochs])*100, 4)) + "%" + "\n"
#        Output += "Final Training Loss: " + str(round(self.__Stats(self.__LossTrainStatistics[self.Sample.Epochs][-1]), 4)) + "" + "\n"
#        Output += "Final Validation Accuracy: " + str(round(self.__Stats(self.__ValidationStatistics[self.Sample.Epochs])*100, 4)) + "%" + "\n"
#        Output += "Final Validation Loss: " + str(round(self.__Stats(self.__LossValidationStatistics[self.Sample.Epochs][-1]), 4)) + "" + "\n\n"
#        
#        Output += "===== Model Information ===== \n"
#        Output += "Learning Rate: " + str(self.Sample.LearningRate) + "\n"
#        Output += "Weight Decay: " + str(self.Sample.WeightDecay) + "\n"
#        Output += "Target Type: " + self.Sample.DefaultTargetType + "\n"
#        Output += "Loss Function: " + self.Sample.DefaultLossFunction + "\n"
#        Output += "Model Name: " + self.Sample.TrainingName + "\n\n"
#
#        def GenericText(_Last, _First, Str1, Str2):
#            Val = self.__Stats(_Last, _First)
#            out = ""
#            if Val < 0:
#                out += Str1 + " Decrease: " + str(round(abs(Val), 4)) + Str2 + " "
#            else:                                                       
#                out += Str1 + " Increase: " + str(round(abs(Val), 4)) + Str2 + " "
#            return out
#
#        Output += "====== Loss Performance From Epoch 1 -> " + str(self.Sample.Epochs) + "====== \n "
#        Delta_LV, Delta_LT = [], []
#        for i in range(self.Sample.Epochs-1):
#            LV_Last, LV_First = self.__LossValidationStatistics[i+1], self.__LossValidationStatistics[i+2]
#            LT_Last, LT_First = self.__LossTrainStatistics[i+1], self.__LossTrainStatistics[i+2]
#
#            Output += "[" + str(i+1) + "/" + str(self.Sample.Epochs) + "]  " + GenericText(LV_Last, LV_First, "Validation Loss", "%") + "|| " + GenericText(LT_Last, LT_First, "Training Loss", "%\n")
#            
#            Delta_LV.append(self.__Stats(LV_Last, LV_First)) 
#            Delta_LT.append(self.__Stats(LT_Last, LT_First))
#       
#
#        Output += GenericText(Delta_LV, "", "---- Average Validation Loss", "% / Epoch \n")
#        Output += GenericText(Delta_LT, "", "---- Average Training Loss", "% / Epoch \n")
#
#        Output += "\n====== Accuracy Performance From Epoch 1 -> " + str(self.Sample.Epochs) + " ====== \n "
#        Delta_LV, Delta_LT = [], []
#        for i in range(self.Sample.Epochs-1):
#            AV_Last, AV_First = self.__ValidationStatistics[i+1], self.__ValidationStatistics[i+2]
#            AT_Last, AT_First = self.__TrainStatistics[i+1], self.__TrainStatistics[i+2]
#
#            Output += "[" + str(i+1) + "/" + str(self.Sample.Epochs) + "] " + GenericText(AV_Last, AV_First, "Validation Accuracy", "%") + "|| " + GenericText(AT_Last, AT_First, "Training Accuracy", "%\n")
#
#            Delta_LV.append(self.__Stats(AV_Last, AV_First)) 
#            Delta_LT.append(self.__Stats(AT_Last, AT_First))
#
#        Output += GenericText(Delta_LV, "", "---- Average Validation Accuracy", "% / Epoch \n")
#        Output += GenericText(Delta_LT, "", "---- Average Training Accuracy", "% / Epoch \n")
#       
#        WriteDirectory().WriteTextFile(Output, Dir, "ModelInformation.txt")
#    
#    def __Stats(self, List1, List2 = ""):
#        def Average(L):
#            if L == 0:
#                return 0.000000000000000000
#            return sum(L)/len(L)
#        
#        if List2 == "":
#            return Average(List1)
#
#        if len(List1) != len(List2):
#            return -1 
#        
#        try:
#            i_s, j_s = [], []
#            for i, j in zip(List1, List2):
#                i_s += i
#                j_s += j
#        except TypeError:
#            i_s, j_s = List1, List2
#
#        res = []
#        for i, j in zip(i_s, j_s):
#            try:
#                res.append(100*(j - i) / j)
#            except ZeroDivisionError:
#                res.append(0)
#        return Average(res)
#
#
#
#
#    def LossTrainingPlot(self, Dir, ErrorBars = False):
#        self.__IsGeneratorType()
#        
#        LossVal = self.__CompilePlot(self.__LossValidationStatistics, "Loss", "Epoch")
#        LossVal.Color = "blue"
#        LossVal.Title = "Validation"
#        LossVal.ErrorBars = ErrorBars
#        
#        LossTra = self.__CompilePlot(self.__LossTrainStatistics, "Loss", "Epoch")
#        LossTra.Color = "Orange"
#        LossTra.Title = "Training"
#        LossTra.ErrorBars = ErrorBars
#
#        ACVal = self.__CompilePlot(self.__ValidationStatistics, "Accuracy (%)", "Epoch")
#        ACVal.Color = "blue"
#        ACVal.Title = "Validation"
#        ACVal.ErrorBars = ErrorBars
#        ACTra = self.__CompilePlot(self.__TrainStatistics, "Accuracy (%)", "Epoch")
#        ACTra.Color = "Orange"
#        ACTra.Title = "Training"
#        ACTra.ErrorBars = ErrorBars
#        
#        C = CombineTGraph()
#        C.Title = "Training and Validation Accuracy with Epoch"
#        C.yMin = 0
#        C.yMax = 120
#        C.Lines = [ACVal, ACTra] 
#        C.CompileLine()
#        C.Save(Dir)
#
#        T = CombineTGraph()
#        T.Title = "Training and Validation Loss Function with Epoch"
#        T.yMin = 0
#        T.yMax = max([max(LossVal.yData), max(LossTra.yData)])*1.5
#        T.Lines = [LossVal, LossTra] 
#        T.CompileLine()
#        T.Save(Dir)
#        
#        self.Receipt(Dir)
#    
#    def __CompilePlot(self, dic, yTitle, xTitle):
#        
#        factor = 1
#        if "Accuracy" in yTitle:
#            factor = 100
#
#        T = TGraph()
#        Epoch = []
#        Loss = []
#        for i in dic:
#            l = dic[i]
#            x = []
#            for k in l:
#                if isinstance(k, list):
#                    for j in k:
#                        x.append(j*factor)
#                else:
#                    x.append(k*factor)
#            Epoch.append(i)
#            Loss.append(x)
#        T.xData = Epoch
#        T.yData = Loss
#        T.yTitle = yTitle
#        T.xTitle = xTitle
#        T.xMin = 0
#        T.xMax = i
#        T.Color = "blue"
#        T.Line()
#        return T
#        
#
#    def __IsGeneratorType(self):
#        if isinstance(self.Sample, GenerateDataLoader):
#            self.__Type += "DATALOADER"
#            if self.Sample.Converted == True:
#                self.__Type += "|CONVERTED"
#
#            if self.Sample.TrainingTestSplit == True:
#                self.__Type += "|TESTSAMPLE"
#
#            if self.Sample.Processed == True:
#                self.__Type += "|PROCESSED"
#        
#        if isinstance(self.Sample, Optimizer):
#            if self.Sample.Loader.Trained == True:
#                self.__Type += "|TRAINED"
#                self.__LossValidationStatistics = self.Sample.LossValidationStatistics
#                self.__LossTrainStatistics = self.Sample.LossTrainStatistics
#                self.__TrainStatistics = self.Sample.TrainStatistics
#                self.__ValidationStatistics = self.Sample.ValidationStatistics
#
