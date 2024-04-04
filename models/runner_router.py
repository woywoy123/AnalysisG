from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Events import *
from AnalysisG.Tools import Tools
from AnalysisG.IO import UpROOT
from AnalysisG import Analysis
from dataset_mapping import DataSets
import production

models = {
    "GRNN" : production.GRNN.RecursiveGraphNeuralNetwork,
}

graphs = {
            "TruthChildren_All" : GraphChildren,
            "TruthChildren_NoNu" : GraphChildrenNoNu,
            "TruthJets_All" : GraphTruthJet,
            "TruthJets_NoNu": GraphTruthJetNoNu,
            "Jets_All" : GraphJet,
            "Jets_NoNu" : GraphJetNoNu,
            "Jets_Detector" : GraphDetector
}


class Models:

    def __init__(self, model_name):
        self.models = models
        self.model_list = list(models)
        msg = "Model not found. Options are: "
        self._this = None
        try: self._this = self.models[model_name]
        except KeyError: print(msg + "-> \n".join(self.model_list))

class Graphs:

    def __init__(self, graph_name):
        self.graphs = graphs
        self.graph_list = list(graphs)
        msg = "Graph not found. Options are: "
        self._this = None
        try: self._this = self.graphs[graph_name]
        except KeyError: print(msg + "-> \n".join(self.graph_list))


class AnalysisBuild:

    def __init__(self, ProjectName = "Project"):
        self.ProjectName = ProjectName
        self.Event = None
        self.Graph = None
        self.DataCache = False
        self.EventCache = False
        self.SamplePath = None
        self.EventStop = None
        self.OutputDir = "./"
        self.Threads = 40

    def AddSampleNameEvent(self, name, n_roots = -1):
        x = UpROOT(self.SamplePath)
        for l, smpl in x.Files.items():
            if name != l.split("/")[-1]: continue
            if n_roots != -1: ls = [l + "/" + i for i in smpl][:n_roots]
            else: ls = [l + "/" + i for i in smpl]

            ana = Analysis()
            ana.ProjectName = self.ProjectName
            ana.OutputDirectory = self.OutputDir
            ana.InputSample(name, ls)
            ana.EventCache = True
            ana.Event = self.Event
            ana.Chunks = 1000
            ana.Threads = self.Threads
            ana.Launch()

    def AddSampleNameGraph(self, algo, name):
        self.Graph = Graphs(algo)._this
        if self.Graph is not None: pass
        else: print("No Graph Implementation"); exit()

        ana = Analysis()
        ana.ProjectName = self.ProjectName
        ana.OutputDirectory = self.OutputDir
        ana.InputSample(name)
        ana.Graph = self.Graph
        ApplyFeatures(ana, algo.split("_")[0])
        ana.EventName = self.Event.__name__
        ana.DataCache = True
        ana.Threads = self.Threads
        ana.Chunks = 10000
        ana.Launch()

    def TrainingSample(self, algo, train_name, training_size = 90):
        self.Graph = Graphs(algo)._this
        ana = Analysis()
        ana.OutputDirectory = self.OutputDir
        ana.ProjectName = self.ProjectName
        ana.TrainingName = train_name
        ana.TrainingSize = training_size
        ana.GraphName = self.Graph.__name__
        ana.DataCache = True
        ana.kFolds = 10
        ana.Threads = 48
        ana.Chunks = 10000
        ana.Launch()

    def ModelTrainer(self, model_name):
        x = Models(model_name)
        if x._this is None: exit()
        return x._this
