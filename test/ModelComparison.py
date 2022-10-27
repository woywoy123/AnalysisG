from AnalysisTopGNN.Model import Model
from AnalysisTopGNN.Tools import Tools, Tables
from AnalysisTopGNN.Samples import Epoch
from AnalysisTopGNN.IO import UnpickleObject

class TrainingComparison:

    def __init__(self):
        self.Epoch = None
        self.ModelStats = {}
        self.MinMetric = {}
        self.MaxMetric = {}
        self.ModelFeatures = {}
        self.Names = []
        self.ModelValues = {}

    def Process(self):
        metrics = []
        self.Names += list(self.ModelStats)

        for name in self.ModelStats:
            metrics += [m for m in list(self.ModelStats[name]) if m not in metrics]

        for m in metrics:
            self.__CompareMetric(m, [self.ModelStats[name][m] for name in self.Names], self.Names)


    def __CompareMetric(self, metric, values, names):
        valu = []
        if isinstance(values[0], dict):
            keys = list(set([t for k in values for t in k]))
            valu = {metric + "_" + str(t) : [k[t][0] for k in values] for t in keys}
        elif isinstance(values[0], list):
            valu = {metric : [v[0] for v in values]}
        else:
            valu = {metric : values}
       
        for m in valu:
            Min_ = min(valu[m])
            Max_ = max(valu[m])
            self.MinMetric[m] = [i for i, v in zip(range(len(valu[m])), valu[m]) if v == Min_]
            self.MaxMetric[m] = [i for i, v in zip(range(len(valu[m])), valu[m]) if v == Max_]
            self.ModelValues[m] = {}
            for model, val in zip(names, valu[m]):
                self.ModelValues[m][model] = val

class ModelComparison(Tools):

    def __init__(self):
        self._Analysis = None
        self._ModelDirectories = {}
        self._ModelSaves = {}
        self.ProjectName = None
        self.Tree = None

    def AddAnalysis(self, analysis):
        if self._Analysis == None:
            self._Analysis = analysis
        else:
            self._Analysis += analysis
    
    def AddModel(self, Name, Directory):
        if Name not in self._ModelDirectories:
            self._ModelDirectories[Name] = []
        Directory = self.AddTrailing(Directory, "/")
        self._ModelDirectories[Name] = [Directory + i for i in self.ls(Directory) if "Epoch-" in i]
  
    
    def __Plots(self, metric):
        #/ Continue here with switch /#
        EpochObj = Epoch()
        if i == "EpochTime":
            cont.Plots = self.EpochTimePlot(epochs, cont.yData, outdir)
            Compile[i] = cont
        
        elif i.startswith("NodeTimes"):
            cont.Plots = self.NodeTimePlot(epochs, cont.yData, outdir, i.split("_")[-1], cont.errData, cont.errData)
            NodeTimes.append(cont)

        elif i.startswith("FoldTime"):
            cont.Plots = self.FoldTimePlot(epochs, cont.yData, outdir, cont.errData, cont.errData)
            Compile[i] = cont

        elif i.startswith("Accuracy"):
            cont.Plots = self.AccuracyPlot(epochs, cont.yData, outdir, i.split("_")[1], i.split("_")[-1], cont.errData, cont.errData)
            Accuracy[i.split("_")[-1]].append(cont)

        elif i.startswith("AUC"):
            cont.Plots = self.AUCPlot(epochs, cont.yData, outdir, i.split("_")[1])
            AUC[i.split("_")[-1]].append(cont)
        
        elif i.startswith("TotalLoss"):
            cont.Plots = self.LossPlot(epochs, cont.yData, outdir, i.split("_")[-1], cont.errData, cont.errData)
            Loss["Total"].append(cont)

        elif i.startswith("Loss_"):
            cont.Plots = self.LossPlot(epochs, cont.yData, outdir, i.split("_")[1], cont.errData, cont.errData)
            Loss[i.split("_")[-1]].append(cont)

    def __CompareTraining(self):
        score = {}
        ep = list(self._TrainingContainer)[0]
        minimize = ["TotalLoss", "Loss", "EpochTime"]
        minimize = {i : 0 for k in minimize for i in self._TrainingContainer[ep].MinMetric if i.startswith(k) }
        maximize = ["AUC", "Accuracy"]
        maximize = {i : 0 for k in maximize for i in self._TrainingContainer[ep].MaxMetric if i.startswith(k) }
        Names = self._TrainingContainer[ep].Names 
        
        Tbl = Tables()
        Tbl.Title = "SUMMARY"
        Tbl.AddColumnTitle("Metrics \ Models")

        for n in Names:
            for k in (list(maximize) + list(minimize)):
                Tbl.AddValues(k.replace("_", " "), n, 0)

        for ep in self._TrainingContainer:
            for m in minimize:
                model = [Names[i] for i in self._TrainingContainer[ep].MinMetric[m]]
                for k in model:
                    Tbl.AddValues(m.replace("_", " "), k, 1)

            for m in maximize:
                model = [Names[i] for i in self._TrainingContainer[ep].MaxMetric[m]]
                for k in model:
                    Tbl.AddValues(m.replace("_", " "), k, 1)

        Tbl.Compile()
        Tbl.DumpTableToFile(self.ProjectName + "/Summary/TrainingComparison")
        epochs = list(self._TrainingContainer)
        epochs.sort()
      
        for i in Names:
            Tbl = Tables()
            Tbl.Title = i 
            Tbl.Sum = False
            Tbl.MinMax = True
            Tbl.AddColumnTitle("Epoch \ Metric")
            for ep in epochs:
                dic = self._TrainingContainer[ep].ModelValues
                for metric in dic:
                    if metric.startswith("NodeTimes"):
                        continue
                    Tbl.AddValues(ep, metric, dic[metric][i])
            
            Tbl.Compile()
            Tbl.DumpTableToFile(self.ProjectName + "/Summary/" + i + "/EpochSummary")
        
        SummedEpoch = {}
        for i in Names:
            SummedEpoch[i] = sum(self._ModelDirectories[i])
            SummedEpoch[i].Compile(self.ProjectName + "/Summary/" + i)

        Plots = {}
        metrics = list(self._TrainingContainer[epochs[0]].ModelValues)
        for met in metrics:
            Plots |= {met : {}}
            for m in Names:
                Plots[met][m] = [self._TrainingContainer[ep].ModelValues[met][m] for ep in epochs]
                


    def Compile(self):
        self._TrainingContainer = {}
        for names in self._ModelDirectories:
            self._ModelSaves[names] = {}
            self._ModelSaves[names]["TorchSave"] = {int(i.split("-")[-1]) : i + "/TorchSave.pth" for i in self._ModelDirectories[names] if self.IsFile(i + "/TorchSave.pth")}
            self._ModelDirectories[names] = [UnpickleObject(i + "/Stats.pkl") for i in self._ModelDirectories[names] if self.IsFile(i+"/Stats.pkl")]
            
            for epch in self._ModelDirectories[names]:
                if epch.Epoch not in self._TrainingContainer:
                    self._TrainingContainer[epch.Epoch] = TrainingComparison()
                self._TrainingContainer[epch.Epoch].Epoch = epch.Epoch
                self._TrainingContainer[epch.Epoch].ModelStats[names] = epch._Package[epch.Epoch]
                self._TrainingContainer[epch.Epoch].ModelFeatures[names] = epch._Package["OutputNames"]
        for ep in self._TrainingContainer:
            self._TrainingContainer[ep].Process()
        self.__CompareTraining()

