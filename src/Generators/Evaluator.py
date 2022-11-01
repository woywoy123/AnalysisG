from AnalysisTopGNN.Model import Model
from AnalysisTopGNN.Tools import Tools, RandomSamplers
from AnalysisTopGNN.Plotting.ModelComparisonPlots import _Comparison, DataBlock, ModelComparisonPlots, PredictionContainer 
from AnalysisTopGNN.Plotting.NodeStatistics import SampleNode
from AnalysisTopGNN.Statistics import Reconstruction
from AnalysisTopGNN.IO import UnpickleObject
from AnalysisTopGNN.Samples import Epoch

class ModelEvaluator(Tools, ModelComparisonPlots, SampleNode, RandomSamplers):

    def __init__(self):
        self._Analysis = None
        self._ModelDirectories = {}
        self._ModelSaves = {}
       
        self._Container = {}
        self._Blocks = {}

        self.ProjectName = None
        self.Tree = None
        self.Device = "cuda"

    def AddAnalysis(self, analysis):
        if self._Analysis == None:
            self._Analysis = analysis
        else:
            self._Analysis += analysis
    
    def AddModel(self, Name, Directory, ModelInstance = None, BatchSize = None):
        if Name not in self._ModelDirectories:
            self._ModelDirectories[Name] = []
        Directory = self.AddTrailing(Directory, "/")
        self._ModelDirectories[Name] = [Directory + i for i in self.ls(Directory) if "Epoch-" in i]
        self._ModelSaves[Name] = {"ModelInstance": ModelInstance} 
        self._ModelSaves[Name] |= {"BatchSize" : BatchSize}

    def __CompareTraining(self, Mode):
        Names = list(self._ModelDirectories)
        outdir = self.ProjectName + "/Summary/" + Mode + "/"
        epochs = list(self._TrainingContainer)
        epochs.sort()
        
        self.Colors = {m : None for m in Names}
        self.GetConsistentModeColor(self.Colors)
        self.Tables(self._TrainingContainer, Mode) 
        
        for met in list(self._TrainingContainer[epochs[0]].ModelValues):
            Plots = []
            for m in Names:
                data = [self._TrainingContainer[ep].ModelValues[met][m] for ep in epochs]
                Err = [self._TrainingContainer[ep].ModelValuesError[met][m] for ep in epochs]
                Lines = self.TemplateLine(m, epochs, data, Err, outdir)
                Lines.Color = self.Colors[m]
                Lines.Compile()
                Plots.append(Lines)
            
            plt = self.PlotTime(Plots, met, outdir) if met.startswith("EpochTime") else None
            plt = self.PlotTime(Plots, met, outdir) if met.startswith("FoldTime") else plt
            plt = self.PlotTime(Plots, met, outdir) if met.startswith("NodeTimes") else plt
            plt = self.PlotAUC(Plots, met, outdir) if met.startswith("AUC") else plt
            plt = self.PlotLoss(Plots, met, outdir) if met.startswith("Loss_") else plt
            plt = self.PlotLoss(Plots, met, outdir) if met.startswith("TotalLoss_") else plt
            plt = self.PlotAccuracy(Plots, met, outdir) if met.startswith("Accuracy") else plt

    def __ShowSampleDistribution(self):
        self.SampleNodes = {}
        self.Training = {}
        self.TestSample = {}
        self.OutDir = self.ProjectName + "/NodeStatistics/"
        self.AddNodeSample(self._Analysis)
        self.Process()

    def __ParticleReconstruction(self, model, sample, smpleprc, container):
        reco = Reconstruction(model)
        idx = 0
        index = 0
        for data in sample:
            prc = smpleprc[index]
            data.to(device = self.Device)
            
            reco.TruthMode = False
            pred = reco(data)

            reco.TruthMode = True
            truth = reco(data) 
            
            for i in range(idx,idx + len(pred)):
                p = self._Analysis.HashToROOT(prc[i-idx]).split("/")[-2]
                rec = {o : reco.ParticleEfficiency(pred[i - idx][o], truth[i - idx][o], p) for o in pred[i-idx]}
                container.ReconstructionEfficiency(rec, truth[i - idx], pred[i-idx])

            idx+= len(pred)
            index += 1
    
    def __EvaluateSample(self, model, smple, Ep):
        for i in smple:
            Ep.EpochContainer.StartTimer()
            pred, truth, loss_acc = model.Prediction(i)
            Ep.EpochContainer.StopTimer()
            Ep.EpochContainer.Collect(pred, truth, loss_acc, Ep.Make)
            
    def __EvaluateSampleType(self, Type):
        for name in self._ModelSaves:
            if self._ModelSaves[name]["ModelInstance"] == None:
                continue
            BatchSize = 1 if self._ModelSaves[name]["BatchSize"] == None else self._ModelSaves[name]["BatchSize"]
            smple, prcsmple = self.MakeSample({i.Filename : i.Trees[self.Tree] for i in self._Analysis if i.Train == Type or Type == None}, True, BatchSize)
            
            model = self._ModelSaves[name]["ModelInstance"] 
            epochsDict = self._ModelSaves[name]["TorchSave"]
            epochs = list(epochsDict) 
            epochs.sort()
            
            if Type:
                make = "Train"
            elif Type == False:
                make = "Test"
            elif Type == None:
                make = "All"

            if make not in self._Container:
                self._Container[make] = {}
            if name not in self._Container[make]:
                self._Container[make][name] = []
        
            for ep in epochs:
                mod = Model(model)
                mod.Device = self.Device
                mod.LoadModel(self.abs(epochsDict[ep]))

                p = PredictionContainer()
                p.ModelName = name
                p.Epoch = ep 
                p.Make = make
                p.EpochContainer = Epoch(ep)
                p.EpochContainer.ModelOutputs += list(mod.GetModelOutputs())
                p.EpochContainer.MakeDictionary(make)
                p.EpochContainer.Fold = 1
                p.EpochContainer.FoldTime[1] = 0
                p.ProjectName = self.ProjectName

                self._Container[make][name].append(p)
                      
                self.__ParticleReconstruction(mod, smple, prcsmple, p)
                self.__EvaluateSample(mod, smple, p)
                
                p.CompileEpoch()

    def __ProcessContainer(self, Container, Name):
        self._TrainingContainer = {}
        ModelContainer = {}
        BlockContainer = {}
        self._Blocks[Name] = DataBlock("Models")
        for model in Container:
            epochs = Container[model]
            ModelContainer[model] = [ep.EpochContainer for ep in epochs]
            self.TrainingComparison(ModelContainer, self._TrainingContainer, model)
            BlockContainer[model] = sum(epochs) 
            BlockContainer[model].CompileMergedEpoch()
            BlockContainer[model] = BlockContainer[model].Plots
        self._Blocks[Name].Plots = BlockContainer
        for ep in self._TrainingContainer:
            self._TrainingContainer[ep].Process()
        self.__CompareTraining(Name)

    def Compile(self):
        self._TrainingContainer = {}
        for names in self._ModelDirectories:
            Dict = {int(i.split("-")[-1]) : i + "/TorchSave.pth" for i in self._ModelDirectories[names] if self.IsFile(i + "/TorchSave.pth")}
            self._ModelSaves[names]["TorchSave"] = Dict
            self._ModelDirectories[names] = [UnpickleObject(i + "/Stats.pkl") for i in self._ModelDirectories[names] if self.IsFile(i+"/Stats.pkl")]
            self.TrainingComparison(self._ModelDirectories, self._TrainingContainer, names)
        
        for ep in self._TrainingContainer:
            self._TrainingContainer[ep].Process()
        self.__CompareTraining("Training")
        self.__ShowSampleDistribution()

        self.__EvaluateSampleType(True) 
        self.__ProcessContainer(self._Container["Train"], "TrainingSample") 
        self.__EvaluateSampleType(False)
        self.__ProcessContainer(self._Container["Test"], "TestSample")        
        self.__EvaluateSampleType(None)
        self.__ProcessContainer(self._Container["All"], "CompleteSample") 
        
        self.Verdict()
