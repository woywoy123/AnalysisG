from TrainingPlots import Training
from AllPlots import All, Train, Test
from Tooling import Template
from LogDump import LogDumper
from AnalysisTopGNN.Plotting import TLine, CombineTLine
   
class FigureContainer:

    def __init__(self):
        self.OutputDirectory = None
        self.training = Training()
        self.all = All()
        self.test = Test()
        self.train = Train()
        self.TrainingPlots = {}
        self.AllPlots = {}
        self.TestPlots = {}
        self.TrainPlots = {}

    def AddEpoch(self, epoch, vals):
        if "training" in vals:
            self.training.AddEpoch(epoch, vals["training"])
        elif "all" in vals:
            self.all.AddEpoch(epoch, vals["all"])
        elif "train" in vals:
            self.train.AddEpoch(epoch, vals["train"])
        elif "test" in vals:
            self.test.AddEpoch(epoch, vals["test"])

    def Compile(self):
        def Convert(inpt):
            for i in inpt:
                if isinstance(inpt[i], dict):
                    for j in inpt[i]:
                        inpt[i][j] = inpt[i][j].DumpDict()
                    continue
                if inpt[i] == None:
                    continue
                inpt[i] = inpt[i].DumpDict()

        self.TrainingPlots |= self.training.Compile(self.OutputDirectory)
        self.AllPlots |= self.all.Compile(self.OutputDirectory, "all")
        self.TestPlots |= self.test.Compile(self.OutputDirectory, "test")
        self.TrainPlots |= self.train.Compile(self.OutputDirectory, "train")
       
        Convert(self.TrainingPlots)
        Convert(self.AllPlots) 
        Convert(self.TestPlots)
        Convert(self.TrainPlots)

        dumped = {
                    "TrainingPlots" : self.TrainingPlots, 
                    "AllPlots" : self.AllPlots, 
                    "TestPlots" : self.TestPlots,
                    "TrainPlots" : self.TrainPlots
                  }
        return dumped

    def Rebuild(self, Dict):
        def GetUniqueID(inpt):
            for i in inpt:
                if "_ID" not in inpt[i]:
                    GetUniqueID(inpt[i])
                    continue
                self._ID[inpt[i]["_ID"]] = inpt[i]
                _id = inpt[i]["Rebuild"] if "Rebuild" in inpt[i] else {}
                self._ID |= _id
        
        def Rebuild(inpt):
            out = {}
            for i in inpt:
                Type = inpt[i]["_TYPE"]
                reco = {k : inpt[i][k] for k in inpt[i] if k not in ["_ID", "_TYPE", "Rebuild", "_temp", "_Varname"]}
                if Type == "TLine":
                    out[i] = TLine(**reco)
                elif Type == "CombineTLine":
                    out[i] = CombineTLine(**reco)
            return out 

        def Assign(inpt):
            for i in inpt:
                if "Rebuild" not in inpt[i]:
                    continue
                
                for _id in inpt[i]["Rebuild"]:
                    v = getattr(self._Out[i], inpt[i]["Rebuild"][_id]["_Varname"])
                    if isinstance(v, list):
                        v += [self._Out[_id]]
                    else:
                        v = self._Out[_id]
                    setattr(self._Out[i], inpt[i]["Rebuild"][_id]["_Varname"], v)
        
        def Place(inpt):
            for i in inpt:
                if "_ID" not in inpt[i]:
                    Place(inpt[i])
                    continue
                _id = inpt[i]["_ID"]
                inpt[i] = self._Out[_id] 

        self.TrainingPlots |= Dict["TrainingPlots"]
        self.AllPlots |= Dict["AllPlots"]

        self._ID = {}
        GetUniqueID(self.AllPlots)
        self._Out = Rebuild(self._ID)
        Assign(self._ID)
        Place(self.AllPlots)

        self._ID = {}      
        GetUniqueID(self.TrainingPlots)
        self._Out = Rebuild(self._ID)
        Assign(self._ID)
        Place(self.TrainingPlots)


class ModelComparison(Template, LogDumper):

    def __init__(self):
        self.TrainingAccuracy = {}
        self.EpochTime = {}

        # ==== Plots for all samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}

        # ==== Plots for test samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}
 
        # ==== Plots for train samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}
 
        self.OutputDirectory = None
        self.Colors = {}
        self._S = " | "
        self._FinalResults = {}

    def AddModel(self, name, Model):
        Model.LoadMergedEpochs()
        self.TrainingAccuracy[name] = Model.Figure.TrainingPlots["Accuracy"]
        self.EpochTime[name] = Model.Figure.TrainingPlots["EpochTime"]

        self.AllAccuracy[name] = Model.Figure.AllPlots["Accuracy"]
        self.AllLoss[name] = Model.Figure.AllPlots["Loss"]
        self.AllAUC[name] = Model.Figure.AllPlots["AUC"]

        self.AllEdgeEffPrc[name] = Model.Figure.AllPlots["EdgeProcessEfficiency"]
        self.AllEdgeEff[name] = Model.Figure.AllPlots["EdgeEfficiency"]

        self.AllNodeEffPrc[name] = Model.Figure.AllPlots["NodeProcessEfficiency"]
        self.AllNodeEff[name] = Model.Figure.AllPlots["NodeEfficiency"]

    def Compare(self, dic, Title, xTitle, yTitle, yMax, Filename):
        com = self.MergePlots(list(dic.values()), self.OutputDirectory)
        for i in dic:
            dic[i].Title = i
        com.xTitle = xTitle
        com.yTitle = yTitle
        com.yMin = -0.1
        com.yMax = yMax
        com.Filename = Filename
        self._FinalResults[(self.OutputDirectory + "/" + Filename).split("ModelComparison/")[-1].replace("/", "-")] = com.Lines
        com.Title = Title
        com.SaveFigure()
        out = self.DumpTLines(com.Lines)
        self.WriteText(out, self.OutputDirectory + "/" + Filename)
        return com

    def CompareEpochTime(self):
        
        com = self.Compare(self.EpochTime, "Time Elapsed at each Epoch", "Epoch", "Time (s)", None, "EpochTime")
        for l in com.Lines:
            self.Colors[l.Title] = l.Color

    def Organize(self, dic):
        lines = list(dic.values())
        names = list(dic)
        
        Features = {}
        for name, l in zip(names, lines):
            Lines = list(l.values()) if isinstance(l, dict) else l.Lines
            for line in Lines:
                feat = line.Title
                if feat not in Features:
                    Features[feat] = {}
                Features[feat][name] = line
                line.Color = self.Colors[name]
        return Features 

    def CompareAccuracy(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Accuracy of Feature " + feat + " Prediction", "Epoch", "Accuracy (%)", 101, prefix + "-" + feat)

    def CompareLoss(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Loss from Predicting " + feat + " Prediction", "Epoch", "Loss (a.u)", None, prefix + "-" + feat)

    def CompareAUC(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Achieved Area under ROC Curve for Feature " + feat, "Epoch", "AUC (Higher is Better)", 1.1, prefix + "-" + feat)

    def CompareReco(self, dic, prefix, prc = ""):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Top Reconstruction Efficiency of Feature " + feat + " Prediction" + prc, "Epoch", "Reconstruction Efficiency (%)", 101, prefix + "-" + feat)
    
    def CompareRecoByProcess(self, dic, prefix):
        Dic = {} 
        for model in dic:
            for feat in dic[model]:
                for p in dic[model][feat].Lines:
                    if p.Title not in Dic:
                        Dic[p.Title] = {}
                    if model not in Dic[p.Title]:
                        Dic[p.Title][model] = {}
                    Dic[p.Title][model][feat] = p
                    p.Title = feat
        tmp = self.OutputDirectory
        for prc in Dic:
            self.OutputDirectory = tmp + "/" + prc
            self.CompareReco(Dic[prc], prefix, " (" + prc +")")

    def Summary(self):
        self.Features = { i : {M : { "Score" : 0, "xData" : [], "yData" : [] } for M in self.Colors} for i in self._FinalResults}

        for feat in self.Features:
            for line in self._FinalResults[feat]:
                if line.DoStatistics:
                    self.Features[feat][line.Title]["xData"] += list(line.xData)
                else:
                    self.Features[feat][line.Title]["xData"] += line.xData
                self.Features[feat][line.Title]["yData"] += line.yData       

        for feat in self.Features:
            perf = {ep : None for ep in self.Features[feat][list(self.Features[feat])[0]]["xData"]}
            ModelName = {ep : [] for ep in self.Features[feat][list(self.Features[feat])[0]]["xData"]}
            for model in self.Features[feat]:
                coef = -1 if sum([1 for met in ["accuracy", "auc", "reco", "EpochTime"] if feat.startswith(met)]) > 0 else 1

                for x, y in zip(self.Features[feat][model]["xData"], self.Features[feat][model]["yData"]):

                    if perf[x] == None:
                        perf[x] = y

                    if perf[x] > y and coef == 1:
                        perf[x] = y
                        ModelName[x] = [model]

                    elif perf[x] < y and coef == -1:
                        perf[x] = y
                        ModelName[x] = [model]

                    elif perf[x] == y:
                        perf[x] = y
                        ModelName[x] += [model]
            
            count = self.UnNestDict(ModelName)
            ModelScore = {m : float(count.count(m)/len(count)) for m in self.Colors}
            for model in ModelScore:
                self.Features[feat][model]["Score"] += ModelScore[model]
                Dict = self.Features[feat][model]
                Min, Max = min(Dict["yData"]), max(Dict["yData"])
                MinIdx, MaxIdx = Dict["yData"].index(Min), Dict["yData"].index(Max)
                Dict["MinMax"] = [Dict["xData"][MinIdx], Min, Dict["xData"][MaxIdx], Max]
        out = self.DumpSummaryTable(self.Features)
        self.WriteText(out, self.OutputDirectory + "/Summary.txt")

    def Compile(self):
        RootDir = self.OutputDirectory + "/ModelComparison"
        AccDir = RootDir + "/accuracy"
        LossDir = RootDir + "/loss"
        AUCDir = RootDir + "/auc"
        RecoEff = RootDir + "/reco"

        self.OutputDirectory = RootDir
        self.CompareEpochTime() 

        self.OutputDirectory = AccDir
        self.CompareAccuracy(self.TrainingAccuracy, "training")
        self.CompareAccuracy(self.AllAccuracy, "all")
            
        self.OutputDirectory = LossDir 
        self.CompareLoss(self.AllLoss, "all")

        self.OutputDirectory = AUCDir
        self.CompareAUC(self.AllAUC, "all")

        self.OutputDirectory = RecoEff
        self.CompareReco(self.AllEdgeEff, "edge-all")
        self.CompareReco(self.AllNodeEff, "node-all")

        self.CompareRecoByProcess(self.AllEdgeEffPrc, "edge-all")
        self.CompareRecoByProcess(self.AllNodeEffPrc, "node-all")


        self.OutputDirectory = RootDir
        self.Summary()

