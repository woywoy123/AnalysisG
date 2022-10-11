from AnalysisTopGNN.IO import UnpickleObject, Directories, WriteDirectory, PickleObject
from AnalysisTopGNN.Generators import EventGenerator, TorchScriptModel, Optimizer, GenerateDataLoader
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Reconstruction import Reconstructor
import os, random
import torch
from sklearn.metrics import roc_curve, auc
from Compilers import *
import numpy as np

class MetricsCompiler(EventGenerator, Optimizer):
    def __init__(self):
        super(MetricsCompiler, self).__init__()
        self.FileEventIndex = {}
        self.Device = None
        self.Threads = 1
        self.CompareToTruth = False
        self.Predictions = {}
        self.Truths = {}
        self.Statistics = {}
        self.model_dict = {}
        self.model_test = {}
        self.ROC_dict = {}
        self.Mass_dict = {}

    def SampleNodes(self, Training, FileTrace):
        self.FileEventIndex |= {"/".join(FileTrace["Samples"][i].split("/")[-2:]) : [FileTrace["Start"][i], FileTrace["End"][i]] for i in range(len(FileTrace["Start"]))}

        RevNode = {ev : [node, True] for node in Training["Training"] for ev in Training["Training"][node]}
        RevNode |= {ev : [node, False] for node in Training["Validation"] for ev in Training["Validation"][node]}
        
        dict_stat = {}
        for evnt in FileTrace["SampleMap"]:
            p_ = self.EventIndexFileLookup(evnt).split("/")[0]
            if p_ not in dict_stat:
                dict_stat[p_] = {"All" : {}, "Training" : {}, "Test" : {} }
            
            n_ = RevNode[evnt][0]
            tr_ = RevNode[evnt][1]
            if n_ not in dict_stat[p_]["All"]:
                dict_stat[p_]["All"][n_] = 0
                dict_stat[p_]["Training"][n_] = 0
                dict_stat[p_]["Test"][n_] = 0

            dict_stat[p_]["All"][n_] += 1
            if tr_:
                dict_stat[p_]["Training"][n_] += 1
            else:
                dict_stat[p_]["Test"][n_] += 1
        return dict_stat


    def UnNest(self, inpt):
        if isinstance(inpt, list) == False:
            return [inpt]
        out = []
        for i in inpt:
            out += self.UnNest(i)
        return out

    def __EpochLoopGeneral(self, Epoch, Statistics, model, m_keys, model_dict, metrics = ["Accuracy", "Loss"]):

        for out in model_dict[model]["Outputs"]:
            for k in [t + "_" + m for t in m_keys for m in metrics]:
                model_dict[model][out][k.replace("_", "")] += self.UnNest(Statistics[k][out])
            for k in m_keys: 
                model_dict[model][out][k + "Epochs"] += [Epoch for i in range(len(self.UnNest(Statistics[k + "_" + metrics[0]][out])))]
    
    def __EpochLoopNodes(self, Epoch, Statistics, model, m_keys, model_dict):
        for n, i in zip(Statistics["Nodes"], range(len(Statistics["Nodes"]))):
            if n not in model_dict[model]["NodeTime"]:
                model_dict[model]["NodeTime"][n] = []
            model_dict[model]["NodeTime"][n] += Statistics["FoldTime"][i]
            
            for out in model_dict[model]["Outputs"]:
                for k in m_keys:
                    if n not in model_dict[model][out]["Node" + k.replace("_", "")]:
                        model_dict[model][out]["Node" + k.replace("_", "")][n] = []
                    model_dict[model][out]["Node" + k.replace("_", "")][n] += Statistics[k][out][i]

    def ModelTrainingCompiler(self, TrainingStatistics):
        GeneralDetails = ["TrainingTime", "Tree", "Start", "End", "Level", "SelfLoop", "Samples", "BatchSize", "Model"]
        for model in TrainingStatistics:
            self.model_dict[model] = {}
            self.model_dict[model] |= { k : TrainingStatistics[model]["Done"][k] for k in GeneralDetails }
            self.model_dict[model]["Outputs"] = list(TrainingStatistics[model]["Done"]["Training_Accuracy"])

            Epochs = [key for key in list(TrainingStatistics[model]) if key != "Done"]
            self.model_dict[model] |= {k : [] for k in ["kFold", "EpochTime", "kFoldTime", "Epochs"]}
            self.model_dict[model]["NodeTime"] = {}
                
            for out in self.model_dict[model]["Outputs"]:
                self.model_dict[model][out] = {}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingLoss", "ValidationLoss"]}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingEpochs", "ValidationEpochs"]}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingAccuracy", "ValidationAccuracy"]}
                self.model_dict[model][out] |= {k : {} for k in ["NodeValidationAccuracy", "NodeTrainingAccuracy"]}
                self.model_dict[model][out] |= {k : {} for k in ["NodeValidationLoss", "NodeTrainingLoss"]}

            for ep in Epochs:
                Statistics = TrainingStatistics[model][ep]
                self.model_dict[model]["EpochTime"] += Statistics["EpochTime"]
                self.model_dict[model]["kFold"] += self.UnNest(Statistics["kFold"])
                self.model_dict[model]["kFoldTime"] += self.UnNest(Statistics["FoldTime"])
                
                self.__EpochLoopGeneral(ep, Statistics, model, ["Training", "Validation"], self.model_dict)
                self.__EpochLoopNodes(ep, Statistics, model, ["Training_Accuracy", "Validation_Accuracy", "Training_Loss", "Validation_Loss"], self.model_dict)
                self.model_dict[model]["Epochs"] += [ep]
        return self.model_dict

    def ModelPrediction(self, TorchSave, Data, TorchScript):
        
        self.Training = False
        self.Debug = "Test"
        for model in TorchSave:
            if model == "Models":
                continue
            BaseModel = torch.load(TorchSave["Models"][model])
            epochs = list(TorchSave[model])
            epochs.sort()
            _zero = epochs.pop(0)

            self._init = False
            self.Sample = Data[0]

            self.Model = BaseModel
            self.Model.load_state_dict(torch.load(TorchSave[model][_zero])["state_dict"])
            
            self.InitializeModel()
            self.Predictions[model] = {}
            self.Predictions[model] |= {ep : {} for ep in epochs}
            
            self.T_Features = {}
            self.GetTruthFlags([], "E")
            self.GetTruthFlags([], "N")
            self.GetTruthFlags([], "G")
            self.Statistics[model] = {}
            self.Statistics[model]["Outputs"] = [k[2:] for k in self.ModelOutputs if k.startswith("O_")]
            for epoch in epochs:

                if model in TorchScript:
                    self.Model = TorchScriptModel(TorchScript[model][1][epoch], maps = TorchScript[model][0])
                    self.Model.to(self.Device)
                else:
                    self.Model.load_state_dict(torch.load(TorchSave[model][_zero])["state_dict"])

                self.Stats = {}
                self.MakeContainer("Test")

                self.Notify("Importing Model: '" + model + "' @ Epoch: " + str(epoch))
                output = {}
                for d in Data:
                    try:
                        truth, pred = self.Train(d)
                    except:
                        self.Warning("Imported TorchScript Failed. Revert to Pickled Model.")
                        self.Model.load_state_dict(torch.load(TorchSave[model][epoch])["state_dict"])
                        self.Model.to(self.Device)
                        truth, pred = self.Train(d) 
                    it = d.i.item()
                    output[it] = pred
                    
                    if self.CompareToTruth and it not in self.Truths:
                        self.Truths[it] = {v : truth[v][0] for v in truth}
                    #self.Predictions[model][epoch] |= output 
                self.Statistics[model][epoch] = self.Stats
                print(self.Truths)
                print(self.Predictions[model][epoch])



    def ModelTestCompiler(self, Statistics):
        for model in Statistics:
            self.model_test[model] = {}
            self.model_test[model]["Outputs"] = Statistics[model]["Outputs"]
            for out in self.model_test[model]["Outputs"]:
                self.model_test[model][out] = {}
                self.model_test[model][out]["TestLoss"] = []
                self.model_test[model][out]["TestAccuracy"] = []
                self.model_test[model][out]["TestEpochs"] = []

            for Epoch in [i for i in list(Statistics[model]) if isinstance(i, int)]:
                Stats = Statistics[model][Epoch]
                self.__EpochLoopGeneral(Epoch, Stats, model, ["Test"], self.model_test)
        return self.model_test

    def ROCCurveCompiler(self, pred, truth, model, features):
        for epoch in pred[model]:
            for f in features:
                if f not in self.ROC_dict:
                    self.ROC_dict[f] = {}
                if epoch not in self.ROC_dict[f]:
                    self.ROC_dict[f][epoch] = {}
                if model not in self.ROC_dict[f][epoch]:
                    self.ROC_dict[f][epoch][model] = { "fpr" : [], "tpr" : [] }
                
                tru, pred_score = [], []
                for ev in pred[model][epoch]:
                    t = truth[ev][f]
                    p = pred[model][epoch][ev][f]
                   
                    p_p = torch.max(torch.softmax(p[1], dim = 1), dim = 1)[0].detach().cpu().numpy()
                    t = t.detach().cpu().numpy()
                    tru.append(t)
                    pred_score.append(p_p)

                tru = np.concatenate(tru)
                pred_score = np.concatenate(pred_score)
                
                fpr, tpr, _ = roc_curve(tru, pred_score)
                auc_ = auc(fpr, tpr) 

                self.ROC_dict[f][epoch][model]["fpr"] += fpr.tolist()
                self.ROC_dict[f][epoch][model]["tpr"] += tpr.tolist()
                self.ROC_dict[f][epoch][model]["auc"] = auc_

    def MassReconstruction(self, TorchSave, Data, Features, EdgeMode = False):
        def IterateOverData(Data, feat, Mode, var_names, Model = None):
            r = Reconstructor(Model = Model)
            out = {}
            for ev in Data:
                val = r(ev).MassFromEdgeFeature(feat, **var_names) if Mode else r(ev).MassFromNodeFeature(feat, **var_names)
                out[ev.i.item()] = val.tolist()
            return out
        
        def GetParticles(pred):
            out = {}
            out["Tops"] = [k for p in pred.values() for k in p]
            return out
        
        def GetProcessParticles(sample_dic, pred):
            out = {}
            out["nProcessTops"] = { p : [] for p in set(sample_dic.values()) }
            for p in pred:
                out["nProcessTops"][sample_dic[p]] += pred[p]
            return out
        
        def Efficiency(truth_proc, pred_proc, sample_dic):
            def ClosestTop(tru, pred):

                res = []
                if len(tru) == 0:
                    return res
                p = pred.pop(0)
                max_tru, min_tru = max(tru), min(tru)
                col = True if p <= max_tru and p >= min_tru else False

                if col == False:
                    if len(pred) == 0:
                        return res
                    return ClosestTop(tru, pred)

                diff = [ abs(p - t) for t in tru ]
                tru.pop(diff.index(min(diff)))
                res += ClosestTop(tru, pred)
                res.append(p)
                return res 

            out = {}
            out["ProcessTruthTops"] = {}
            out["ProcessPredTops"] = {}
            out["EventEfficiency"] = []
            out["nTrueTops"] = 0
            out["nPredTops"] = 0

            for i in pred_proc:
                pred_tops = pred_proc[i]
                tru_tops = truth_proc[i]
                proc = sample_dic[i]
                
                if proc not in out["ProcessTruthTops"]:
                    out["ProcessTruthTops"][proc] = 0
                    out["ProcessPredTops"][proc] = 0

                p_, t_ = [], []
                p_ += pred_tops
                t_ += tru_tops

                ntops = ClosestTop(t_, p_)
                l_pred = len(ntops)
                l_tru = len(tru_tops)
                
                out["ProcessPredTops"][proc] += l_pred
                out["ProcessTruthTops"][proc] += l_tru
                out["nTrueTops"] += l_tru
                out["nPredTops"] += l_pred
                out["EventEfficiency"] += [float(l_pred / l_tru) * 100]
            return out



        dic = self.Mass_dict
        for feat in Features:
            var_names = Features[feat]

            if "Process" not in dic and self.CompareToTruth:
                dic["Process"] = {}
                for ev in Data:
                    dic["Process"][ev.i.item()] = self.EventIndexFileLookup(ev.i.item()).split("/")[0]

            if feat not in dic:
                dic[feat] = {}
            if self.CompareToTruth and "Truth" not in dic[feat]:
                truth = IterateOverData(Data, feat, EdgeMode, var_names)
                dic[feat]["Truth"] = {}
                dic[feat]["Truth"] |= GetParticles(truth)

            for model in TorchSave:
                Epochs = list(TorchSave[model])
                Epochs.sort()
                for epoch in Epochs:
                    if epoch not in dic[feat]:
                        dic[feat][epoch] = {}
                    if model not in dic[feat][epoch]:
                        dic[feat][epoch][model] = {}
                        
                    self.Notify("Using Model: '" + model + "' @ Epoch: " + str(epoch) + " and Feature: " +  feat)
                    Pred = IterateOverData(Data, feat, EdgeMode, var_names, torch.load(TorchSave[model][epoch]))
                    dic[feat][epoch][model] |= GetParticles(Pred)

                    if self.CompareToTruth == False:
                        continue
                    dic[feat][epoch][model] |= Efficiency(truth, Pred, dic["Process"])
                    




class ModelEvaluator(EventGenerator, Directories, WriteDirectory, GenerateDataLoader):
    
    def __init__(self):
        super(ModelEvaluator, self).__init__()
        self._rootDir = None
        self.Caller = "ModelEvaluator"
        self.Device = "cpu"
        self.DataCacheDir = "HDF5"
        self.Threads = 10
        self.chnk = 100

        # ==== Compiler Classes
        self._MetricsCompiler = MetricsCompiler()
        self._LogCompiler = LogCompiler()
        self._GraphicsCompiler = GraphicsCompiler() 

        # ==== Internal Sample Information
        self._DataLoaderMap = {}
        self._SampleDetails = {}
        self._TestSample = {}
        self._TrainingSample = {}
        self._SamplesHDF5 = []
        
        # ==== HDF5 Stuff 
        self.RebuildSize = 100
        self.RebuildRandom = True

        # ==== Nodes Compiler 
        self._CompileSampleNodes = False
        self.MakeSampleNodesPlots = True

        # ==== Model Compiler
        self.MakeStaticHistogramPlot = True
        self.MakeTrainingPlots = True
        self.MakeTestPlots = True
        self.CompareToTruth = False
        self._CompileModelOutput = False
        self._CompileModels = False
        
        self._ROCCurveFeatures = {}
        self._TrainingStatistics = {}
        self._TorchScripts = {}
        self._TorchSave = {"Models" : {}}

        # ===== Mass Reconstructor ===== #
        self._MassEdgeFeature = {}
        self._MassNodeFeature = {}

    def AddFileTraces(self, Directory):
        if Directory.endswith("FileTraces.pkl"):
            Directory = "/".join(Directory.split("/")[:-1])
        if Directory.endswith("FileTraces") == False:
            Directory = Directory + "/FileTraces"
        self._rootDir = "/".join(Directory.split("/")[:-1])
        self._CompileSampleNodes = True

    def AddModel(self, Directory):
        if Directory.endswith("/"):
            Directory = Directory[:-1]
        Name = Directory.split("/")[-1]

        tmp = self.VerboseLevel
        self.VerboseLevel = 0
        x = self.ListFilesInDir(Directory + "/Statistics/", [".pkl"])
        if len(x) == 0:
            self.Warning("Model: " + Name + " not found. Skipping")
            return 
        self._TrainingStatistics[Name] = x
        files = self.ListFilesInDir(Directory + "/TorchSave/", [".pt"])
        self._TorchSave["Models"][Name] = [i for i in files if "_Model" in i][0]
        self._TorchSave[Name] = {int(k.split("/")[-1].split("_")[1].split(".")[0]) : k for k in files if "_Model" not in k}
        
        self.VerboseLevel = tmp
        self.Notify("Added Model: " + Directory.split("/")[-1])
        self._CompileModelOutput = True

    def AddTorchScriptModel(self, Name, OutputMap = None, Directory = None):
        if Directory is None:
            Directory = self._rootDir + "/Models/" + Name + "/TorchScript/"
        self._TorchScripts[Name] = self.ListFilesInDir(Directory, [".pt"])
        self._TorchScripts[Name] = {int(k.split("/")[-1].split("_")[1].split(".")[0]) : k for k in self._TorchScripts[Name]}
        if len(self._TorchScripts[Name]) == 0:
            self.Warning(Name + " not found. Skipping")
            return 
        if OutputMap == None:
            pt = TorchScriptModel(self._TorchScripts[Name][0])
            pt.ShowNodes()
            self.Fail("OutputMap is not defined. Showing Output Nodes.")
        elif isinstance(OutputMap, list) == False:
            self.Fail("To assign the output, you need to provide a dictionary list. E.g. [{ 'name' : '...', 'node' : '...' }]")
        else:
            pt = TorchScriptModel(self._TorchScripts[Name][0], VerboseLevel = 3, maps = OutputMap)
        
        for i in self._TorchScripts:
            self._TorchScripts[i] = [OutputMap, self._TorchScripts[i]]
        self._CompileModelOutput = True

    def ROCCurveFeature(self, feature):
        self._ROCCurveFeatures[feature] = None

    def MassFromEdgeFeature(self, feature, pt_name = "N_pT", eta_name = "N_eta", phi_name = "N_phi", e_name = "N_energy"):
        self._MassEdgeFeature[feature] = {"pt" : pt_name, "eta" : eta_name, "phi" : phi_name, "e" : e_name}

    def MassFromNodeFeature(self, feature, pt_name = "N_pT", eta_name = "N_eta", phi_name = "N_phi", e_name = "N_energy"):
        self._MassNodeFeature[feature] = {"pt" : pt_name, "eta" : eta_name, "phi" : phi_name, "e" : e_name}

    def __BuildSymlinks(self):
        tmp = self.VerboseLevel 
        self.VerboseLevel = 0
        lst = list(self._DataLoaderMap)
        self.VerboseLevel = tmp

        if self.RebuildSize:
            random.shuffle(lst) 
        
        pkl = list(self.ListDirs(self._rootDir + "/DataCache"))
        pkl = {str(val) : di.split("/")[-1] for di in pkl for val in list(UnpickleObject(di + "/" + di.split("/")[-1] + ".pkl").values())}
        
        self.MakeDir(self._rootDir + "/HDF5")
        leng = int(len(lst)*(self.RebuildSize/100))
        for i in range(leng):
            name = self._DataLoaderMap[lst[i]].split("/")[-1]
            try:
                self._SamplesHDF5.append(name)
                src = self._rootDir + "/DataCache/" + pkl[name] + "/" + self.EventIndexFileLookup(lst[i]).split("/")[-1] + "/" + name + ".hdf5"
                os.symlink(src, os.path.abspath(self._rootDir + "/HDF5/" + name + ".hdf5"))
            except FileExistsError:
                pass
            self.Notify("!!!Creating Symlink: " + name)
            if (i+1) % 10000 == 0 or self.VerboseLevel == 3: 
                self.Notify("!!" + str(round(float(i/leng)*100, 3)) + "% Complete")
        self.DataCacheDir = self._rootDir + "/" + self.DataCacheDir + "/"

    def Compile(self, OutDir = "./ModelEvaluator"):

        if self._CompileSampleNodes:
            FileTrace = UnpickleObject(self._rootDir + "/FileTraces/FileTraces.pkl")
            keys = ["Tree", "Start", "End", "Level", "SelfLoop", "Samples"]
            self._SampleDetails |= { key : FileTrace[key] for key in keys}
            self._DataLoaderMap |= FileTrace["SampleMap"]
            self.FileEventIndex = {"/".join(FileTrace["Samples"][i].split("/")[-2:]).replace(".root", "") : [FileTrace["Start"][i], FileTrace["End"][i]] for i in range(len(FileTrace["Start"]))}

            TrainSample = UnpickleObject(self._rootDir + "/FileTraces/TrainingSample.pkl")
            self._TrainingSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Training"][node]] for node in  TrainSample["Training"] }
            self._TestSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Validation"][node]] for node in TrainSample["Validation"] }
            #dict_stat = self._MetricsCompiler.SampleNodes(TrainSample, FileTrace)

            self._GraphicsCompiler.pwd = OutDir
            self._GraphicsCompiler.MakeSampleNodesPlot = self.MakeSampleNodesPlots
            #self._GraphicsCompiler.SampleNodes(dict_stat)
            self._LogCompiler.pwd = OutDir
            #self._LogCompiler.SampleNodes(dict_stat)

            self.__BuildSymlinks()
        if len(self._SamplesHDF5) == 0:
            self.Fail("No Samples Found.")
        Data = self.RecallFromCache(self._SamplesHDF5, self._rootDir + "/" + self.DataCacheDir) 
        if self._CompileModels:
            EpochContainer = {}
            for model in self._TrainingStatistics:
                EpochContainer[model] = {k.split("/")[-1].split("_")[1].split(".")[0] : UnpickleObject(k) for k in self._TrainingStatistics[model]}

            #model_dict = self._MetricsCompiler.ModelTrainingCompiler(EpochContainer)
            self._GraphicsCompiler.MakeStaticHistograms = self.MakeStaticHistogramPlot
            self._GraphicsCompiler.MakeTrainingPlots = self.MakeTrainingPlots
            
            #for model in model_dict:
            #    self._GraphicsCompiler.pwd = OutDir + "/" + model
            #    self._GraphicsCompiler.TrainingPlots(model_dict, model)
        
        if self._CompileModelOutput and self.MakeTestPlots:
            self._MetricsCompiler.Device = self.Device
            self._MetricsCompiler.CompareToTruth = self.CompareToTruth
            
            self._MetricsCompiler.ModelPrediction(self._TorchSave, Data, self._TorchScripts) 
            stat = self._MetricsCompiler.Statistics
            
            stat_dict = self._MetricsCompiler.ModelTestCompiler(stat)
            for model in stat_dict:
                self._GraphicsCompiler.pwd = OutDir + "/" + model 
                self._GraphicsCompiler.TestPlots(stat_dict, model)
        
        if len(self._ROCCurveFeatures) > 0 or len(self._TorchSave) > 0 and self.CompareToTruth:

            truth = self._MetricsCompiler.Truths
            pred = self._MetricsCompiler.Predictions
            for model in pred:
                features = [f for f in self._ROCCurveFeatures if f in stat[model]["Outputs"]]
                if len(features) == 0:
                    continue
                self._MetricsCompiler.ROCCurveCompiler(pred, truth, model, features)
            ROC_val = self._MetricsCompiler.ROC_dict
            self._GraphicsCompiler.ROCCurve(ROC_val)

        if len(self._MassEdgeFeature) > 0 or len(self._TorchSave) > 0:
            
            self._MetricsCompiler.CompareToTruth = self.CompareToTruth
            self._MetricsCompiler.MassReconstruction(self._TorchSave, Data, self._MassEdgeFeature, True)
            mass_dict = self._MetricsCompiler.Mass_dict

            self._GraphicsCompiler.pwd = OutDir + "/" 
            self._GraphicsCompiler.ParticleReconstruction(mass_dict)
