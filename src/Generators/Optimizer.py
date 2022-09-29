import torch
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch_geometric.loader import DataLoader
from torchmetrics.functional import accuracy
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi

from sklearn.model_selection import KFold
import numpy as np

import time
import datetime
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.IO import WriteDirectory, Directories, PickleObject, ExportToDataScience, UnpickleObject
from AnalysisTopGNN.Generators import GenerateDataLoader, ModelImporter
from AnalysisTopGNN.Parameters import Parameters

class Optimizer(ExportToDataScience, GenerateDataLoader, ModelImporter, Parameters):

    def __init__(self, DataLoaderInstance = None):
        self.Caller = "OPTIMIZER"
        self.Notification()
        self.Optimizer()
        self._init = False
 
        ### DataLoader Inheritence 
        if DataLoaderInstance != None:
            self.ReadInDataLoader(DataLoaderInstance)       

    def ReadInDataLoader(self, DataLoaderInstance):
        self.TrainingSample = DataLoaderInstance.TrainingSample

        self.EdgeAttribute = DataLoaderInstance.EdgeAttribute
        self.NodeAttribute = DataLoaderInstance.NodeAttribute
        self.GraphAttribute = DataLoaderInstance.GraphAttribute

        self.Device = DataLoaderInstance.Device
        self.FileTraces = DataLoaderInstance.FileTraces 

    def LoadLastState(self):
        if self.RunDir == None:
            OutDir = self.RunName
        else:
            OutDir = self.RunDir + "/" + self.RunName

        out = []
        for i in Directories().ListFilesInDir(OutDir + "/TorchSave"):
            out.append(int(i.split("/")[-1].split("_")[1]))
        out.sort()
        if len(out) == 0:
            self.StartEpoch = 0
            return 
        self.StartEpoch = out[-1]

        state = torch.load(OutDir + "/TorchSave/Epoch_" + str(self.StartEpoch) + "_" + str(self.Epochs) + ".pt")
        self.Model.load_state_dict(state["state_dict"])
        self.Optimizer.load_state_dict(state["optimizer"])
        self.StartEpoch = state["epoch"]

        self.Model = torch.load(OutDir + "/TorchSave/Epoch_" + str(self.StartEpoch) + "_" + str(self.Epochs) + ".pt")
        self.Model.train()

    def DumpStatistics(self):
        if self.Debug == True:
            return 
        if self.RunDir == None:
            OutDir = self.RunName
        else:
            OutDir = self.RunDir + "/" + self.RunName

        WriteDirectory().MakeDir(OutDir + "/Statistics")
        if self.epoch == "Done":
            PickleObject(self.Stats, "Stats_" + self.epoch, OutDir + "/Statistics")
        else:
            PickleObject(self.Stats, "Stats_" + str(self.epoch+1), OutDir + "/Statistics")
        self.__MakeStats()

    def MakeContainer(self, Mode):
        self.Stats[Mode + "_Accuracy"] = {}
        self.Stats[Mode + "_Loss"] = {}
        for i in self.T_Features:
            self.Stats[Mode + "_Accuracy"][i] = []
            self.Stats[Mode + "_Loss"][i] = []
    
    def __MakeStats(self):

        ### Output Information
        self.Stats = {}
        self.Stats["EpochTime"] = []
        self.Stats["kFold"] = []
        self.Stats["FoldTime"] = []
        self.Stats["Nodes"] = []

        self.MakeContainer("Training")
        self.MakeContainer("Validation")

    def DefineOptimizer(self):
        if self.DefaultOptimizer == "ADAM":
            self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        elif self.DefaultOptimizer == "SGD":
            self.Optimizer = torch.optim.SGD(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
    
    def DefineScheduler(self):
        if self.DefaultScheduler == "ExponentialLR":
            self.Scheduler = ExponentialLR(self.Optimizer, *self.SchedulerParams)
        elif self.DefineScheduler == None:
            self.Scheduler = None
        elif self.DefineScheduler == "CyclicLR":
            self.Scheduler = CyclicLR(self.Optimizer, *self.SchedulerParams)

    def GetTruthFlags(self, Input = [], FEAT = ""):
        if len(Input) == 0:
            Input = list(self.Sample.to_dict())
            Input = [i.replace(FEAT+"_", "") for i in Input if i.startswith(FEAT + "_")]
        for i in Input:
            if i.startswith("T_") and str("O_" + i[2:]) in self.ModelOutputs:
                self.T_Features[i[2:]] = [FEAT + "_" +i, FEAT + "_" +i[2:]]

    def SampleLoop(self, samples):
        for i in samples:
            self.Train(i) 

    def KFoldTraining(self):
        def CalcAverage(Mode, k, Notify = "", Mode2 = None):
            for f_k in self.Stats[Mode]:
                if self.Debug:
                    continue
                v_1 = str(format(float(sum(self.Stats[Mode][f_k][k])/len(self.Stats[Mode][f_k][k])), ' >10.4g'))
                v_2 = str(format(float(sum(self.Stats[Mode2][f_k][k])/len(self.Stats[Mode2][f_k][k])), ' >10.4g'))
                    
                self.Notify(Notify + "       " + str(v_1) + " || " + str(v_2) + " (" + f_k + ")")
        
        if self.Model == None:
            self.Fail("No Model has been given!")

        self.DefineOptimizer()
        N_Nodes = list(self.TrainingSample)
        N_Nodes.sort(reverse = True)
        self.Sample = self.RecallFromCache(self.TrainingSample[N_Nodes[0]][0], self.CacheDir)

        self.InitializeModel()
        self.Notify(">------------------------ Starting k-Fold Training ------------------------------------")
        self.Notify("!SIZE OF ENTIRE SAMPLE SET: " + str(sum([len(self.TrainingSample[i]) for i in N_Nodes])))
        self.GetTruthFlags(self.EdgeAttribute, "E")
        self.GetTruthFlags(self.NodeAttribute, "N")
        self.GetTruthFlags(self.GraphAttribute, "G")

        if len(self.T_Features) == 0:
            self.Fail("NO TRUTH FEATURES WERE FOUND DURING INITIALIZATION PHASE!")
        self.Notify(">----------------------------------------------------------------------------------------\n")
        
        self.LoadLastState()
        self.Model.Device = str(self.Device)
        
        if self.DefaultScheduler != None:
            self.DefineScheduler()
        
        for i in self.TrainingSample:
            self.TrainingSample[i] = self.RecallFromCache(self.TrainingSample[i], self.CacheDir)

        # ===== Make a preliminary Dump of the Sample being trained on ====== #
        self.__MakeStats()
        self.Stats["TrainingTime"] = 0 
        self.Stats.update(self.FileTraces)
        
        self.Stats["n_Node_Files"] = [[] for i in range(len(self.Stats["Start"]))]
        self.Stats["n_Node_Count"] = [[] for i in range(len(self.Stats["Start"]))]
        TMP = [smpl for node in self.TrainingSample for smpl in self.TrainingSample[node]]       
        for s_i in range(len(TMP)):
            smpl = TMP[s_i]
            indx, n_nodes = int(smpl.i), int(smpl.num_nodes)
            find = [indx >= int(start) and indx <= int(end) for start, end in zip(self.Stats["Start"], self.Stats["End"])].index(True)
            if n_nodes not in self.Stats["n_Node_Files"][find]:
                self.Stats["n_Node_Files"][find].append(n_nodes)
                self.Stats["n_Node_Count"][find].append(0)
            
            n_i = self.Stats["n_Node_Files"][find].index(n_nodes)
            self.Stats["n_Node_Count"][find][n_i] += 1
     
        self.Stats["BatchSize"] = self.BatchSize
        self.Stats["Model"] = {}
        self.Stats["Model"]["LearningRate"] = self.LearningRate
        self.Stats["Model"]["WeightDecay"] = self.WeightDecay
        self.Stats["Model"]["ModelFunctionName"] = str(type(self.Model))
        self.Stats["Model"]["Scheduler"] = self.DefaultScheduler
        self.Stats["Model"]["SchedulerParams"] = str(self.SchedulerParams)
        self.Stats["Model"]["Optimizer"] = self.DefaultOptimizer
        self.epoch = "Done"
        self.DumpStatistics()
        # ===== Make a preliminary Dump of the Sample being trained on ====== #

        TimeStart = time.time()
        for self.epoch in range(self.StartEpoch, self.Epochs):
            self.Notify("! >============== [ EPOCH (" + str(self.epoch+1) + "/" + str(self.Epochs) + ") ] ==============< ")
            
            Splits = KFold(n_splits = self.kFold, shuffle = True, random_state= 42)
            TimeStartEpoch = time.time()
            k = 0
            for n_node in N_Nodes:
                Curr = self.TrainingSample[n_node] 
                Curr_l = len(Curr)

                self.Notify("!+++++++++++++++++++++++")
                self.Notify("!NUMBER OF NODES -----> " + str(n_node) + " NUMBER OF ENTRIES: " + str(Curr_l))

                if Curr_l < self.kFold:
                    self.Warning("NOT ENOUGH SAMPLES FOR EVENTS WITH " + str(n_node) + " PARTICLES :: SKIPPING")
                    continue

                self.Stats["FoldTime"].append([])
                self.Stats["kFold"].append([])
                for fold, (train_idx, val_idx) in enumerate(Splits.split(np.arange(Curr_l))):

                    for f in self.T_Features:
                        self.Stats["Training_Accuracy"][f].append([])
                        self.Stats["Validation_Accuracy"][f].append([])
                        self.Stats["Training_Loss"][f].append([])
                        self.Stats["Validation_Loss"][f].append([])

                    TimeStartFold = time.time()
                    self.Notify("!!CURRENT k-Fold: " + str(fold+1))

                    self.Training = True
                    train_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(train_idx))
                    self.SampleLoop(train_loader)
                    
                    self.Notify("!!!-----++> Training <++-----")
                    self.Notify("!!!-------> Accuracy || Loss  <-------")
                    CalcAverage("Training_Accuracy", k, "!!!", "Training_Loss")

                    self.Training = False
                    valid_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(val_idx)) 
                    self.SampleLoop(valid_loader)
                    self.Notify("!!!-----==> Validation <==-----")
                    self.Notify("!!!-------> Accuracy || Loss  <-------")
                    CalcAverage("Validation_Accuracy", k, "!!!", "Validation_Loss")

                    self.Stats["FoldTime"][-1].append(time.time() - TimeStartFold)
                    self.Stats["kFold"][-1].append(fold+1)
                   
                    k += 1
                self.Stats["Nodes"].append(n_node)

                self.Notify("!! >----- [ EPOCH (" + str(self.epoch+1) + "/" + str(self.Epochs) + ") ] -----< ")
                self.Notify("!!-------> Training || Validation <-------")
                self.Notify("!!_______________ Accuracy/MSE _______________")
                CalcAverage("Training_Accuracy", k-1, "!!", "Validation_Accuracy")
                self.Notify("!!_______________ LOSS _______________")
                CalcAverage("Training_Loss", k-1, "!!", "Validation_Loss")
                #torch.cuda.empty_cache()
            
            self.Stats["EpochTime"].append(time.time() - TimeStartEpoch)
            self.Notify("! >========= DURATION: " + str(datetime.timedelta(seconds = self.Stats["EpochTime"][-1])))
            self.DumpStatistics()
            
            if self.Scheduler != None:
                self.Scheduler.step()

            self.ExportModel(valid_loader)
       
        if self.RunDir == None:
            OutDir = self.RunName
        else:
            OutDir = self.RunDir + "/" + self.RunName

        self.Stats = UnpickleObject("Stats_Done", OutDir + "/Statistics")
        self.Stats["TrainingTime"] = time.time() - TimeStart
        self.epoch = "Done"
        self.DumpStatistics()


    def Train(self, sample):
        if self.Training:
            self.Model.train()
            self.Optimizer.zero_grad()
            self._mode = "Training"
        else:
            self.Model.eval()
            self._mode = "Validation"

        self.MakePrediction(sample)
        truth_out = self.Output(self.ModelOutputs, sample, Truth = True)
        model_out = self.Output(self.ModelOutputs, sample, Truth = False)
            
        if self.Debug:
            longest = max([len(i) for i in model_out])
        
        LT = 0
        for key in model_out:
            t_p = truth_out[key][0]
            m_p = model_out[key][0]
            
            t_v = truth_out[key][1]
            m_v = model_out[key][1]
            
            Loss = self.ModelOutputs["L_" + key]
            Classification = False
            acc = None
            if Loss == "CEL":
                t_v = t_v.type(torch.LongTensor).view(-1).to(self.Device)
                t_p = t_p.type(torch.int)
                acc = accuracy
                self.LF = torch.nn.CrossEntropyLoss()
                Classification = True
                
            elif Loss == "MSEL":
                t_v = t_v.type(torch.float).to(self.Device)
                self.LF = torch.nn.MSELoss()
                acc = self.LF

            elif Loss == "HEL":
                self.LF = torch.nn.HingeEmbeddingLoss()
                t_v = t_v.type(torch.LongTensor).to(self.Device)
                acc = self.LF

            elif Loss == "KLD":
                self.LF = torch.nn.KLDivLoss()
                t_v.type(torch.LongTensor)
                m_v.type(torch.LongTensor)
                acc = self.LF
            
            # Error Handling if something goes wrong. Can cause CUDA to completely freeze Python
            # Can be solved by changing screen resolution....
            if acc == None:
                self.Warning("SKIPPING " + key + " :: NO LOSS FUNCTION SELECTED!!!!")
                continue

            elif m_v.shape[1] == 1 and self.ModelOutputs["C_" + key]:
                pass

            elif m_v.shape[1] <= t_v.max() and (self.ModelOutputs["C_" + key] or Classification):
                self.Fail("(" + key + ") Your Classification Model only has " 
                        + str(int(m_v.shape[1])) + " classes but requires " + str(int(t_v.max()+1)))
            elif Classification == False and m_v.shape[1] != t_v.shape[1]:
                self.Warning("Model is using regression, but your truth has length " 
                        + str(int(t_v.shape[1])) + " but need " + str(int(m_v.shape[1])))
                self.Fail("Your Model has more outputs than Truth! :: " + key)
            acc = acc(m_p, t_p)
            L = self.LF(m_v, t_v)
            LT += L
            
            if self.Debug == True:
                print("----------------------(" + key + ") -------------------------------")
                print("---> Truth: \n", t_p.tolist())
                print("---> Model: \n", m_p.tolist())
                print("---> DIFF: \n", (t_p - m_p).tolist())
                print("(Loss)---> ", float(L))
            elif self.Debug == "Loss":
                dif = key + " "*int(longest - len(key))
                print(dif + " | (Loss)---> ", float(L))
            elif self.Debug == "Pred":
                print("---> Truth: \n", t_p.tolist())
                print("---> Model: \n", m_p.tolist())
            elif self.Debug == "Test":
                self.Stats["Test_Accuracy"][key].append(acc.item())
                self.Stats["Test_Loss"][key].append(L.item())
                continue
            elif self.Debug:
                continue
            
            self.Stats[self._mode + "_Accuracy"][key][-1].append(acc.item())
            self.Stats[self._mode + "_Loss"][key][-1].append(L.item())

        if self.Training:
            LT.backward()
            self.Optimizer.step()
        
        if self.Debug:
            return truth_out, model_out

