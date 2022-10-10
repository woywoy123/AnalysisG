from AnalysisTopGNN.Tools import Notification
from Tooling import Tools 
from Samples import SampleContainer
from Models import ModelContainer
from Figures import ModelComparison


class ModelEvaluator(Tools, Notification):
    
    def __init__(self):
        self._rootDir = self.pwd()
        self._Models = {}
        self.Caller = "ModelEvaluator"
        self.Device = "cuda"
        self.VerboseLevel = 3
        
        self.Threads = 12
        self.chnks = 20
        self.BuildDataPercentage = 10

        self.BuildDataRandom = False
        self.BuildData = True
        self.MakePlotsOnly = False

        self.MergeEpochs = True
        self.EpochMax = None
        self.EpochMin = 0

    def AddFileTraces(self, Directory):
        if Directory.endswith("/"):
            Directory = Directory[:-1]
        self._rootDir = Directory

    def AddModel(self, Directory):
        if Directory.endswith("/"):
            Directory = Directory[:-1]
        ModelName = Directory.split("/")[-1]
        self._Models[ModelName] = ModelContainer(ModelName)
        self._Models[ModelName].Dir = Directory
        self._Models[ModelName].Device = self.Device
        self._Models[ModelName].VerboseLevel = self.VerboseLevel
        self._Models[ModelName].Threads = self.Threads
        self._Models[ModelName].chnks = self.chnks
    
    def __AddFeatureToModel(self, obj, name, dic, feat):
        inter = getattr(obj, name)
        if feat not in inter:
            inter[feat] = {"ROC" : False, "Mass" : False}
        inter[feat] |= dic 

    def DefineTorchScriptModel(self, Name, OutputNodeMap):
        self._Models[Name].TorchScriptMap = OutputNodeMap

    def MassEdgeFeature(self, Feature, pt_name = "N_pT", eta_name = "N_eta", phi_name = "N_phi", e_name = "N_energy"):
        for i in self._Models:
            dic = {"varnames" : {"pt" : pt_name, "eta" : eta_name, "phi" : phi_name, "e" : e_name}, "Mass" : True}
            self.__AddFeatureToModel(self._Models[i], "EdgeFeatures", dic, Feature)

    def MassNodeFeature(self, Feature, pt_name = "N_pT", eta_name = "N_eta", phi_name = "N_phi", e_name = "N_energy"):
        for i in self._Models:
            dic = {"varnames" : {"pt" : pt_name, "eta" : eta_name, "phi" : phi_name, "e" : e_name}, "Mass" : True}
            self.__AddFeatureToModel(self._Models[i], "NodeFeatures", dic, Feature)

    def ROCEdgeFeature(self, Feature):
        for i in self._Models:
            self.__AddFeatureToModel(self._Models[i], "EdgeFeatures", {"ROC" : True}, Feature)

    def ROCNodeFeature(self, Feature):
        for i in self._Models:
            self.__AddFeatureToModel(self._Models[i], "NodeFeatures", {"ROC" : True}, Feature)

    def ROCGraphFeature(self, Feature):
        for i in self._Models:
            self.__AddFeatureToModel(self._Models[i], "GraphFeatures", {"ROC" : True}, Feature)


    def Compile(self, OutputDirectory):
        if self.MakePlotsOnly == False:
            self.mkdir(OutputDirectory + "/HDF5")
            DataContainer = SampleContainer()
            DataContainer.Device = self.Device
            DataContainer.random = self.BuildDataRandom
            DataContainer.Size = self.BuildDataPercentage
            DataContainer.DataCache = self.abs(self._rootDir + "/DataCache")
            DataContainer.FileTrace = self._rootDir + "/FileTraces/FileTraces.pkl"
            DataContainer.TrainingSample = self._rootDir + "/FileTraces/TrainingSample.pkl"
            DataContainer.HDF5 = self.abs(OutputDirectory + "/HDF5")
            DataContainer.Collect(self.BuildData)
            DataContainer.MakeSamples()
            DataContainer.Compile()
            
            Sample = list(DataContainer.SampleMap.values())[0].Data
            Data = DataContainer.SampleMap
            
            for i in self._Models:
                self._Models[i].Collect()
                self._Models[i].MakeEpochs(self.EpochMin, self.EpochMax)
                self._Models[i].OutputDirectory = OutputDirectory
                
                self._Models[i].AnalyzeDataCompatibility(Sample)

                self._Models[i].CompileTrainingStatistics()
                self._Models[i].CompileResults("test", Data)
                self._Models[i].CompileResults("train", Data)
                self._Models[i].CompileResults("all", Data)
                self._Models[i].Purge()
        
        M = ModelComparison()
        M.OutputDirectory = OutputDirectory
        for i in self._Models:
            self._Models[i].OutputDirectory = OutputDirectory
            if self.MergeEpochs:
                self._Models[i].MergeEpochs()
            M.AddModel(i, self._Models[i]) 
        M.Compile() 

