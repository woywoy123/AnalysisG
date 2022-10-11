from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Generators import Optimizer
from Tooling import Tools, Metrics
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

class Epoch(Tools, Optimizer, Metrics):

    def __init__(self):
        self.Epoch = None
        self.ModelName = None       
        self.Model = None
        self.ModelBase = None
        self.TorchSave = None
        self.TorchScript = None
        self.ONNX = None
        self.TrainStats = None
        self.ModelInputs = None
        self.ModelOutputs = None
        self.ModelTruth = None
        self.Training = False
        self.Device = None
        self.BatchSize = None
        
        self.Debug = "Test"
        self.Stats = {}
        self.ROC = {}
        
        self.NodeFeatureMass = {}
        self.EdgeFeatureMass = {}

        self.TruthNodeFeatureMass = {}
        self.TruthEdgeFeatureMass = {}

        self.NodeParticleEfficiency = {}
        self.EdgeParticleEfficiency = {}

    def CollectMetric(self, name, key, feature, inpt):
        if name not in self.Stats:
            self.Stats[name] = {}
        if feature not in self.Stats[name]:
            self.Stats[name][feature] = []
        for i in self.TraverseDictionary(inpt, key):
            self.Stats[name][feature] += self.UnNestList(i)

    def CompileTraining(self):
        self.TrainStats = UnpickleObject(self.TrainStats)
        self.Stats["EpochTime"] = self.TrainStats["EpochTime"][0]
        Metrics = ["Training_Loss", "Training_Accuracy", "Validation_Loss", "Validation_Accuracy"]
        for metric in Metrics:
            for feat in self.TrainStats[metric]:
                self.CollectMetric(metric.replace("_", ""), metric + "/" + feat, feat, self.TrainStats)
      
        self.Stats["FoldTime"] = []
        self.Stats["KFolds"] = []
        for n_node in range(len(self.TrainStats["Nodes"])):
            node = self.TrainStats["Nodes"][n_node]
            self.Stats["FoldTime"] += self.TrainStats["FoldTime"][n_node]
            self.Stats["KFolds"] += self.TrainStats["kFold"][n_node]
           
            kF = len(self.TrainStats["kFold"][n_node])
            for metric in Metrics:
                for feat in self.TrainStats[metric]:
                    inpt = self.TrainStats[metric][feat][n_node*kF:(n_node+1)*kF]
                    self.CollectMetric("Node"+metric.replace("_", ""), self.TrainStats[metric][feat], str(node) + "/" + feat, inpt)
        del self.TrainStats
   
    def LoadModel(self):
        model = torch.load(self.ModelBase)
        state = torch.load(self.TorchSave)
        model.load_state_dict(state["state_dict"])
        self.Model = model
        self.Model.eval()

    def PredictOutput(self, DataInpt, idx):
        data = [D.Data for D in DataInpt]
        prc = [[D.prc] for D in DataInpt]
        inpt = next(iter(DataLoader(data, batch_size = len(idx))))
        truth, pred = self.Train(inpt)

        PredDataStruc = {}
        PredDataStruc |= {feat : pred[feat][0].view(-1, 1) for feat in list(pred)}
        PredDataStruc |= {feat + "_score" : pred[feat][1] for feat in list(pred)}
        PredDataStruc["edge_index"] = inpt.edge_index
        PredDataStruc["batch"] = inpt.batch.view(-1, 1)
        pred = Data().from_dict(PredDataStruc)

        TruthDataStruc = {feat : truth[feat][1] for feat in list(truth)}
        TruthDataStruc["edge_index"] = inpt.edge_index
        TruthDataStruc["batch"] = inpt.batch.view(-1, 1)
        truth = Data().from_dict(TruthDataStruc)
        
        ModelOut = [i[2:] for i in self.ModelOutputs if i.startswith("O_")]
        pred = [pred.subgraph(inpt.batch == b) for b in range(len(idx))]
        truth = [truth.subgraph(inpt.batch == b) for b in range(len(idx))]
        for feat in ModelOut:
            if feat not in self.ROC:
                self.ROC[feat] = { "fpr" : [], "tpr" : [], "auc" : [], 
                                   "truth" : [], "pred" : [], "pred_score" : [], 
                                   "idx" : [], "proc" : []
                                 } 

            for b in range(len(idx)):
                self.ROC[feat]["truth"].append(truth[b][feat].view(-1))
                self.ROC[feat]["pred"].append(pred[b][feat].view(-1))
                self.ROC[feat]["pred_score"].append(pred[b][feat + "_score"]) 

                self.ROC[feat]["idx"] += [idx[b]]
                self.ROC[feat]["proc"] += prc[b]
        
        return pred, ModelOut

    def Flush(self):
        self.Stats = {}
        self.ROC = {}
        self.NodeFeatureMass = {}
        self.EdgeFeatureMass = {}
        self.TruthNodeFeatureMass = {}
        self.TruthEdgeFeatureMass = {}
        self.NodeParticleEfficiency = {}
        self.EdgeParticleEfficiency = {}
        self.MakeContainer(self.Debug)

    def ParticleYield(self, Edge):
        dic_p = self.EdgeFeatureMass if Edge else self.NodeFeatureMass
        dic_t = self.TruthEdgeFeatureMass if Edge else self.TruthNodeFeatureMass
        dic_o = self.EdgeParticleEfficiency if Edge else self.NodeParticleEfficiency

        for pack in [[feat, event, prc] for feat in dic_p for event, prc in zip(dic_p[feat], self.ROC[feat]["proc"])]:
            f, idx, prc = pack[0], pack[1], pack[2]
            if f not in dic_o:
                dic_o[f] = {}
            dic_o[f][idx] = self.ParticleEfficiency(dic_p[f][idx], dic_t[f][idx], prc)
    
    def DumpEpoch(self, ModeType, OutputDir):
        def ParticleDumping(Pred_Mass, Truth_Mass, Effic, Key):
            Output = {}
            for i in Pred_Mass:
                Title = Key + "ParticleMass/"+ i + "/MassDistribution"
                Output[Title] = {"Truth" : self.UnNestDict(Truth_Mass[i]), "Prediction" : self.UnNestDict(Pred_Mass[i])}
                
                prc = self.CollectKeyNestDict(Effic[i], "Prc")
                per = self.CollectKeyNestDict(Effic[i], "%")
                Title = Key + "ParticleMass/"+ i + "/ProcessEfficiency"
                Output[Title] = {p : [ per[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
    
                ntru = self.CollectKeyNestDict(Effic[i], "ntru")
                Title = Key + "ParticleMass/"+ i + "/SampleComposition"
                Output[Title] = {p + "Truth" : [ ntru[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
    
                pred = self.CollectKeyNestDict(Effic[i], "nrec")
                Output[Title] |= {p + "Predicted" : [ pred[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
   
                Title = Key + "ParticleMass/" + i + "/AllCollectedParticles"
                Output[Title] = (float(sum(pred)/sum(ntru)))*100
            return Output

        def ROCDumping(ROC_Dict):
            Output = {}
            Title = "ROC/CombinedFeatures"
            Output[Title] = {}
            Output["AUC/AllCollected"] = {}
            for feat in ROC_Dict:
                Output[Title] |= { 
                                    feat : {"FPR" : ROC_Dict[feat]["fpr"], "TPR" : ROC_Dict[feat]["tpr"]} 
                                }
                Output["AUC/AllCollected"] |= {feat : ROC_Dict[feat]["auc"]}
            return Output

        def DumpSampleLoss(Feat, LossDict):
            return {"Loss/" + Feat : LossDict[Feat]}

        def DumpSampleAccuracy(Feat, AccDict):
            return {"Accuracy/" + Feat : AccDict[Feat]}
        
        self.mkdir(OutputDir + "/" + self.ModelName + "/" + ModeType + "/Epochs/")
        Out = {}
        if ModeType == "training": 
            Out |= self.Stats
            self.Stats = {}
        else:
            for metric in self.Stats:
                for feat in self.Stats[metric]:
                    Out |= DumpSampleAccuracy(feat, self.Stats[metric]) if "_Accuracy" in metric else DumpSampleLoss(feat, self.Stats[metric])
            
            Out |= ParticleDumping(self.EdgeFeatureMass, self.TruthEdgeFeatureMass, self.EdgeParticleEfficiency, "Edge")
            Out |= ParticleDumping(self.NodeFeatureMass, self.TruthNodeFeatureMass, self.NodeParticleEfficiency, "Node")
            Out |= ROCDumping(self.ROC)
        
        self.Flush()
        PickleObject(Out, str(self.Epoch), OutputDir + "/" + self.ModelName + "/" + ModeType + "/Epochs/") 
