from Tooling import Tools 
from Epoch import Epoch
from Figures import * 
from AnalysisTopGNN.Generators import ModelImporter
from AnalysisTopGNN.Tools import Notification, Threading
from AnalysisTopGNN.Reconstruction import Reconstructor
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
import torch

class ModelContainer(Tools, Reconstructor):

    def __init__(self, Name = None):
        self.Epochs = {}
        self.TorchScriptMap = None
        self.ModelSaves = {
            "TorchSave" : {}, 
            "TorchScript" : {}, 
            "Base" : None, 
            "Training" : {}
        }
        
        self.EdgeFeatures = {}
        self.NodeFeatures = {}
        self.GrapFeatures = {}
        self.T_Features = {}
        
        self.ModelOutputs = {}
        self.ModelInputs = {} 

        self.Name = Name
        self.Dir = None
        self.VerboseLevel = 0
        self.Threads = None
        self.chnks = None

        self.Caller = "ModelContainer"
        self.OutputDirectory = None

    def Collect(self):
        Files = self.ListFilesInDir(self.Dir + "/TorchSave")
        
        self.ModelSaves["Base"] = Files.pop(Files.index([i for i in Files if "_Model.pt" in i][0]))
        self.Epochs |= { int(ep.split("_")[1]) : None for ep in Files}
        self.ModelSaves["TorchSave"] |= { ep.split("_")[1] : self.Dir + "/PickleSave/" + ep for ep in Files} 

        Files = self.ListFilesInDir(self.Dir + "/TorchScript")
        self.ModelSaves["TorchScript"] |= { ep.split("_")[1] : self.Dir + "/TorchScript/" + ep for ep in Files} 
        
        Files = self.ListFilesInDir(self.Dir + "/Statistics")
        self.ModelSaves["Training"] |= { ep.split("_")[1].replace(".pkl", "") : self.Dir + "/Statistics/" + ep for ep in Files}

        TrainParams = UnpickleObject(self.ModelSaves["Training"]["Done"])
        self.TrainingParameters = {}
        self.TrainingParameters |= {"BatchSize" : TrainParams["BatchSize"]} 
        self.TrainingParameters |= TrainParams["Model"]
        self.BatchSize = self.TrainingParameters["BatchSize"]

    def MakeEpochs(self, Min = 0, Max = None):
        self.SortEpoch(self.Epochs)
        self.Epochs = {i : self.Epochs[i] for i in list(self.Epochs)[Min : Max]}
        for ep in self.Epochs:
            self.Epochs[ep] = Epoch()
            self.Epochs[ep].BatchSize = self.BatchSize
            self.Epochs[ep].Epoch = ep
            self.Epochs[ep].ModelName = self.Name
            self.Epochs[ep].TrainStats = self.ModelSaves["Training"][str(ep)]
            self.Epochs[ep].TorchSave = self.ModelSaves["TorchSave"][str(ep)]
            self.Epochs[ep].TorchScript = self.ModelSaves["TorchScript"][str(ep)]
            self.Epochs[ep].ModelBase = self.Dir + "/TorchSave/" + self.ModelSaves["Base"]
            self.Epochs[ep].Device = self.Device
    
    def AnalyzeDataCompatibility(self, Sample):
        self._init = False
        self.Model = torch.load(self.Dir + "/TorchSave/" + self.ModelSaves["Base"])
        self.Sample = Sample
        self.InitializeModel()
        self.GetTruthFlags(FEAT = "E")
        self.GetTruthFlags(FEAT = "N")
        self.GetTruthFlags(FEAT = "G")
        self.Model = None 
        self.Sample = None
        for i in self.Epochs:
            self.Epochs[i].ModelInputs = self.ModelInputs
            self.Epochs[i].ModelOutputs = self.ModelOutputs
            self.Epochs[i].T_Features = self.T_Features

    def CompileTrainingStatistics(self):
        for ep in self.Epochs:
            self.Epochs[ep].CompileTraining()
            self.Epochs[ep].DumpEpoch("training", self.OutputDirectory) 

    def RebuildParticles(self, Features, Edge, idx):
       for i in Features:
            if Features[i]["Mass"] == False:
                continue
            truthkey = "E_T_" + i if Edge and self.TruthMode else i
            truthkey = "N_T_" + i if Edge == False and self.TruthMode else truthkey
            if truthkey not in self.Sample and i not in self.T_Features:
                continue
            mass_dic = self.EdgeFeatureMass if Edge else self.NodeFeatureMass
            if i not in mass_dic:
                mass_dic[i] = {}

            m = self.MassFromEdgeFeature(i, **Features[i]["varnames"]).tolist() if Edge else []
            m = self.MassFromNodeFeature(i, **Features[i]["varnames"]).tolist() if Edge == False else m
            mass_dic[i][idx] = m

    def CompileResults(self, sample, DataOriginal):
        switch = True if sample == "test" else False
        switch = False if sample == "train" else switch
        switch = None if sample == "all" else switch

        self.TruthMode = True
        self.EdgeFeatureMass = {}
        self.NodeFeatureMass = {}
        Data = {idx : DataOriginal[idx] for idx in DataOriginal if DataOriginal[idx].train != switch }
        
        for idx in Data:
            self._Results = Data[idx].Data
            self.Sample = Data[idx].Data
            self.RebuildParticles(self.EdgeFeatures, True, idx)
            self.RebuildParticles(self.NodeFeatures, False, idx)
        
        for ep in self.Epochs:
            self.Epochs[ep].Flush()
            
            self.Epochs[ep].TruthEdgeFeatureMass |= self.EdgeFeatureMass
            self.Epochs[ep].TruthNodeFeatureMass |= self.NodeFeatureMass

        DataIdx, DataBatch = list(Data), list(Data.values())
        DataIdx = [DataIdx[i : i+self.BatchSize] for i in range(0, len(DataIdx), self.BatchSize)]
        DataBatch = [DataBatch[i : i+self.BatchSize] for i in range(0, len(DataBatch), self.BatchSize)]

        self.TruthMode = False
        for ep in self.Epochs:

            self.EdgeFeatureMass = {}
            self.NodeFeatureMass = {}
            self.Epochs[ep].LoadModel()
            
            for data, idx in zip(DataBatch, DataIdx):
                pred, modelOuts = self.Epochs[ep].PredictOutput(data, idx)
                
                for b in range(len(idx)):
                    self.Sample = data[b].Data
                    self._Results = { "O_" + i : pred[b][i] for i in modelOuts}
                    self.RebuildParticles(self.EdgeFeatures, True, idx[b])
                    self.RebuildParticles(self.NodeFeatures, False, idx[b])
                
            self.Epochs[ep].NodeFeatureMass |= self.NodeFeatureMass
            self.Epochs[ep].EdgeFeatureMass |= self.EdgeFeatureMass
            self.Epochs[ep].ParticleYield(True)
            self.Epochs[ep].ParticleYield(False)

            for feat in self.EdgeFeatures:
                if feat not in self.Epochs[ep].ROC: 
                    continue
                if self.EdgeFeatures[feat]["ROC"]:
                    self.Epochs[ep].MakeROC(feat)
            
            for feat in self.NodeFeatures:
                if feat not in self.Epochs[ep].ROC: 
                    continue
                if self.NodeFeatures[feat]["ROC"]:
                    self.Epochs[ep].MakeROC(feat)
            self.Epochs[ep].DumpEpoch(sample, self.OutputDirectory)
            self.Notify("(" + self.Name + ") DUMPED EPOCH: " + str(ep) + " WITH SAMPLE: " + sample)
        self._Results = None
        self.Sample = None

    def Purge(self):
        for i in self.Epochs:
            val = self.Epochs[i]
            del val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def MergeEpochs(self):
        def Function(inpt):
            for i in range(len(inpt)):
                out = inpt[i]
                inpt[i] = [int(out[1]), {out[0] :  UnpickleObject(out[2])}]
            return inpt
        
        
        ModelDir = self.OutputDirectory + "/" + self.Name
        self.Figure = FigureContainer()
        self.Figure.OutputDirectory = ModelDir

        Epochs = []
        Modes = self.ListFilesInDir(ModelDir)
        for mode in Modes:
            for pkl in self.ListFilesInDir(ModelDir + "/" + mode + "/Epochs/"):
                Epochs.append([mode, int(pkl.replace(".pkl", "")), ModelDir + "/" + mode + "/Epochs/" + pkl])

        TH = Threading(Epochs, Function, self.Threads, self.chnks)
        TH.Start()
        for c in TH._lists:
            self.Figure.AddEpoch(c[0], c[1])  
        dump = self.Figure.Compile()
        PickleObject(dump, "Epochs", ModelDir)


    def LoadMergedEpochs(self):
        Dict = UnpickleObject("Epochs", self.OutputDirectory + "/" + self.Name)
        self.Figure = FigureContainer()
        self.Figure.Rebuild(Dict)

