from .TemplateHistograms import TH1F, CombineTH1F
from .TemplateLines import TLine, CombineTLine
from AnalysisTopGNN.Tools import Tools, Tables

class _Comparison:

    def __init__(self):
        self.Epoch = None
        self.ModelStats = {}
        self.MinMetric = {}
        self.MaxMetric = {}
        self.ModelFeatures = {}
        self.Names = []
        self.ModelValues = {}
        self.ModelValuesError = {}

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
            valu = {metric + "_" + str(t) : [k[t][0] if t in k else 0 for k in values] for t in keys}
            errs = {metric + "_" + str(t) : [k[t][1] if t in k else 0 for k in values] for t in keys}

        elif isinstance(values[0], list):
            valu = {metric : [v[0] for v in values]}
            errs = {metric : [v[1] for v in values]}

        else:
            valu = {metric : values}
            errs = {metric : [0 for m in values]}
       
        for m in valu:
            Min_ = min(valu[m])
            Max_ = max(valu[m])
            
            self.MinMetric[m] = [i for i, v in zip(range(len(valu[m])), valu[m]) if v == Min_]
            self.MaxMetric[m] = [i for i, v in zip(range(len(valu[m])), valu[m]) if v == Max_]
            
            self.ModelValues[m] = {}
            self.ModelValuesError[m] = {}
            
            for model, val, err in zip(names, valu[m], errs[m]):
                self.ModelValues[m][model] = val
                self.ModelValuesError[m][model] = err

class Plots:

    def __init__(self):
        pass

    def GetConsistentModeColor(self, Color):
        Plt = CombineTLine()
        Line = TLine()
        for i in Color:
            Plt.ApplyRandomColor(Line)
            Color[i] = Line.Color
            Line.Color = None

    def TemplateLine(self, Title, xData, yData, err, outdir):
        Plots = {
                    "xTitle" : "Epoch",
                    "Title" : Title,
                    "xData" : xData, 
                    "yData" : yData, 
                    "up_yData" : err, 
                    "down_yData" : err, 
                    "yMin" : 0,
                    "xMax" : max(xData) + 1 if len(xData) > 0 else None,
                    "xMin" : min(xData) - 1 if len(xData) > 0 else None,
                    "yMax" : max([k+i for k, i in zip(err, yData)]) if len(err) > 0 else None,
                    "Style" : "ATLAS",
                    "OutputDirectory" : outdir 
                }
        if len(yData) != 0:
            return TLine(**Plots)
        else:
            return Plots
    
    def PlotTime(self, figs, metric, outdir):
        Plots = self.TemplateLine(metric.replace("_", " "), figs[0].xData, [], [], outdir + "Time")
        Plots["yTitle"] = "Time in Seconds (Lower is Better)"
        Plots["Lines"] = figs
        Plots["Filename"] = metric
        Plots["yMax"] = max([k+i for t in figs for k, i in zip(t.up_yData, t.yData)])
        com = CombineTLine(**Plots)
        com.SaveFigure()
    
    def PlotAUC(self, figs, metric, outdir):
        Plots = self.TemplateLine(metric.replace("_", " "), figs[0].xData, [], [], outdir + "AUC")
        Plots["yTitle"] = "Area under ROC Curve (Higher is Better)"
        Plots["Lines"] = figs
        Plots["Filename"] = metric
        Plots["yMax"] = 1
        com = CombineTLine(**Plots)
        com.SaveFigure()
    
    def PlotLoss(self, figs, metric, outdir):
        Plots = self.TemplateLine(metric.replace("_", " "), figs[0].xData, [], [], outdir + "Loss")
        Plots["yTitle"] = "Loss of Model (a.u) (Lower is Better)"
        Plots["Lines"] = figs
        Plots["Filename"] = metric
        Plots["yMax"] = max([k+i for t in figs for k, i in zip(t.up_yData, t.yData)])
        com = CombineTLine(**Plots)
        com.SaveFigure()
 
    def PlotAccuracy(self, figs, metric, outdir):
        Plots = self.TemplateLine(metric.replace("_", " "), figs[0].xData, [], [], outdir + "Accuracy")
        Plots["yTitle"] = "Loss of Accuracy (%) (Higher is Better)"
        Plots["Lines"] = figs
        Plots["Filename"] = metric
        Plots["yMax"] = 100 
        com = CombineTLine(**Plots)
        com.SaveFigure()
 

class ModelComparisonPlots(Plots):

    def __init__(self):
        pass

    def Tables(self, Container):
        ep = list(Container)[0]
        minimize = ["TotalLoss", "Loss", "EpochTime"]
        minimize = {i : 0 for k in minimize for i in Container[ep].MinMetric if i.startswith(k) }
        maximize = ["AUC", "Accuracy"]
        maximize = {i : 0 for k in maximize for i in Container[ep].MaxMetric if i.startswith(k) }
        Names = Container[ep].Names 
        
        Tbl = Tables()
        Tbl.Title = "SUMMARY"
        Tbl.AddColumnTitle("Metrics \ Models")

        for n in Names:
            for k in (list(maximize) + list(minimize)):
                Tbl.AddValues(k.replace("_", " "), n, 0)

        for ep in Container:
            for m in minimize:
                model = [Names[i] for i in Container[ep].MinMetric[m]]
                for k in model:
                    Tbl.AddValues(m.replace("_", " "), k, 1)

            for m in maximize:
                model = [Names[i] for i in Container[ep].MaxMetric[m]]
                for k in model:
                    Tbl.AddValues(m.replace("_", " "), k, 1)

        Tbl.Compile()
        Tbl.DumpTableToFile(self.ProjectName + "/Summary/TrainingComparison")
        epochs = list(Container)
        epochs.sort()
      
        for i in Names:
            Tbl = Tables()
            Tbl.Title = i 
            Tbl.Sum = False
            Tbl.MinMax = True
            Tbl.AddColumnTitle("Epoch \ Metric")
            for ep in epochs:
                dic = Container[ep].ModelValues
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
    
    def TrainingComparison(self, ModelResults, TrainContainer, names):
        for epch in ModelResults[names]:
            if epch.Epoch not in TrainContainer:
                TrainContainer[epch.Epoch] = _Comparison()
            TrainContainer[epch.Epoch].Epoch = epch.Epoch
            TrainContainer[epch.Epoch].ModelStats[names] = epch._Package[epch.Epoch]
            TrainContainer[epch.Epoch].ModelFeatures[names] = epch._Package["OutputNames"]
 
