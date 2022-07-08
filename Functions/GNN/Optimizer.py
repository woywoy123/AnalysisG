import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import accuracy
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi

from sklearn.model_selection import KFold
import numpy as np

import time
import datetime
from Functions.Tools.Alerting import Notification
from Functions.IO.Files import WriteDirectory, Directories
from Functions.IO.IO import PickleObject
from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.DataLoader import GenerateDataLoader

class ModelImporter:

    def __init__(self, ExampleSample):
        Notification.__init__(self)
        self.Caller = "MODELIMPORTER"
        self.Sample = None
        self.InitializeModel()
        
    def InitializeModel(self):
        self.Model.to(self.Device)
        self.ModelInputs = list(self.Model.forward.__code__.co_varnames[:self.Model.forward.__code__.co_argcount])
        self.ModelInputs.remove("self")
        input_len = len(list(set(list(self.Sample.to_dict())).intersection(set(self.ModelInputs))))
        
        if input_len < len(self.ModelInputs):
            
            self.Warning("---> ! Features Expected in Model Input ! <---")
            for i in self.ModelInputs:
                self.Warning("+-> " + i)
            
            self.Warning("---> ! Features Found in Sample Input ! <---")
            for i in list(self.Sample.__dict__["_store"]):
                self.Warning("+-> " + i)
            self.Fail("MISSING VARIABLES IN GIVEN DATA SAMPLE")

        self.Notify("FOUND ALL MODEL INPUT PARAMETERS IN SAMPLE")
        for i in self.ModelInputs:
            self.Notify("---> " + i)

        self.Notify("AVAILABLE PARAMETERS FOUND IN SAMPLE")
        for i in list(self.Sample.__dict__["_store"]):
            self.Notify("---> " + i)

        self.ModelOutputs = {i : k for i, k in self.Model.__dict__.items() for p in ["C_", "L_", "O_"] if i.startswith(p)}
        self.ModelOutputs |= {i : None for i, k in self.Model.__dict__.items() if i.startswith("O_")}

    def MakePrediction(self, sample):
        dr = {}
        for i in self.ModelInputs:
            dr[i] = sample[i]
        self.Model(**dr)

    def Output(self, output_dict, sample, Truth = False):
        def GetKeyPair(dic, key):
            if key in dic:
                return dic[key]
            else:
                return False
        OutDict = {} 
        for key in output_dict:
            if key.startswith("O_") == False:
                continue

            key = key.lstrip("O_")
            if Truth: 
                out_v = sample[self.T_Features[key][0]]
            else:
                out_v = self.Model.__dict__["O_" + key]
            out_p = out_v

            # Adjust the outputs
            if GetKeyPair(output_dict, "C_" + key) and not Truth:
                out_p = out_v.max(dim = 1)[1]
            out_p = out_p.view(1, -1)[0]
            OutDict[key] = [out_p, out_v]
        return OutDict

class Optimizer(ExportToDataScience, GenerateDataLoader, ModelImporter, Notification):

    def __init__(self, DataLoaderInstance = None):
        self.Verbose = True
        Notification.__init__(self, self.Verbose)

        self.Caller = "OPTIMIZER"
        ### DataLoader Inheritence 
        if DataLoaderInstance != None:
            self.ReadInDataLoader(DataLoaderInstance)

        ### User defined ML parameters
        self.LearningRate = 0.0001
        self.WeightDecay = 0.001
        self.kFold = 10
        self.Epochs = 10
        self.BatchSize = 10
        self.Model = None
        self.RunName = "UNTITLED"
        self.RunDir = "_Models"
        self.DefaultOptimizer = "ADAM"
        self.ONNX_Export = False
        self.TorchScript_Export = True
        self.Debug = False
        
        ### Internal Stuff 
        self.Training = True
        self.T_Features = {}
        self.CacheDir = None

    def ReadInDataLoader(self, DataLoaderInstance):
        self.TrainingSample = DataLoaderInstance.TrainingSample

        self.EdgeAttribute = DataLoaderInstance.EdgeAttribute
        self.NodeAttribute = DataLoaderInstance.NodeAttribute
        self.GraphAttribute = DataLoaderInstance.GraphAttribute

        self.Device = DataLoaderInstance.Device
        self.FileTraces = DataLoaderInstance.FileTraces 

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

    def __MakeStats(self):

        ### Output Information
        self.Stats = {}
        self.Stats["EpochTime"] = []
        self.Stats["BatchRate"] = []
        self.Stats["kFold"] = []
        self.Stats["FoldTime"] = []
        self.Stats["Nodes"] = []

        self.Stats["Training_Accuracy"] = {}
        self.Stats["Validation_Accuracy"] = {}
        self.Stats["Training_Loss"] = {}
        self.Stats["Validation_Loss"] = {}

        for i in self.T_Features:
            self.Stats["Training_Accuracy"][i] = []
            self.Stats["Validation_Accuracy"][i] = []
            self.Stats["Training_Loss"][i] = []
            self.Stats["Validation_Loss"][i] = []


    def DefineOptimizer(self):
        if self.DefaultOptimizer == "ADAM":
            self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        elif self.DefaultOptimizer == "SGD":
            self.Optimizer = torch.optim.SGD(self.Model.parameters(), lr = self.LearningRate)

    def Train(self, sample):
        if self.Training:
            self.Model.train()
            self.Optimizer.zero_grad()
        else:
            self.Model.eval()

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
                t_v = t_v.type(torch.LongTensor).to(self.Device).view(1, -1)[0]
                t_p = t_p.type(torch.int)
                acc = accuracy
                self.LF = torch.nn.CrossEntropyLoss()
                Classification = True
                
            elif Loss == "MSEL":
                t_v = t_v.type(torch.float).to(self.Device)
                self.LF = torch.nn.MSELoss()
                acc = self.LF
            
            # Error Handling if something goes wrong. Can cause CUDA to completely freeze Python
            if acc == None:
                self.Warning("SKIPPING " + key + " :: NO LOSS FUNCTION SELECTED!!!!")
                continue
            if m_v.shape[1] <= t_v.max() and (self.ModelOutputs["C_" + key] or Classification):
                self.Fail("(" + key + ") Your Classification Model only has " + str(int(m_v.shape[1])) + " classes but requires " + str(int(t_v.max()+1)))
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
            if self.Debug:
                continue

            if self.Training:
                self.Stats["Training_Accuracy"][key][-1].append(acc)
                self.Stats["Training_Loss"][key][-1].append(L)
            else:
                self.Stats["Validation_Accuracy"][key][-1].append(acc)
                self.Stats["Validation_Loss"][key][-1].append(L)

        if self.Training:
            LT.backward()
            self.Optimizer.step()
        
        if self.Debug:
            return truth_out, model_out


    def SampleLoop(self, samples):
        self.ResetAll() 
        self.len = len(samples.dataset)
        R = []
        
        for i in samples:
            if self.Training:
                self.ProgressInformation("TRAINING")
            else:
                self.ProgressInformation("VALIDATING")
            self.Train(i) 
            R.append(self.Rate)
        if self.AllReset:
            self.Stats["BatchRate"].append(R)

    def GetTruthFlags(self, Input, FEAT):
        for i in Input:
            if i.startswith("T_") and str("O_" + i[2:]) in self.ModelOutputs:
                self.T_Features[i[2:]] = [FEAT + "_" +i, FEAT + "_" +i[2:]]

    def KFoldTraining(self):
        def CalcAverage(Mode, k, Notify = "", Mode2 = None):
            for f_k in self.Stats[Mode]:
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
        self.Notify(">----------------------------------------------------------------------------------------\n")

        self.__MakeStats()
        self.Model.Device = str(self.Device)
        
        TimeStart = time.time()
        for self.epoch in range(self.Epochs):
            self.Notify("! >============== [ EPOCH (" + str(self.epoch+1) + "/" + str(self.Epochs) + ") ] ==============< ")
            
            Splits = KFold(n_splits = self.kFold, shuffle = True, random_state= 42)
            TimeStartEpoch = time.time()
            k = 0
            for n_node in N_Nodes:
                Curr = self.RecallFromCache(self.TrainingSample[n_node], self.CacheDir)
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
                    memory_rem = get_gpu_memory_from_nvidia_smi()[0]
                    self.SampleLoop(train_loader)
                    self.Notify("!!!-----++> Training <++-----")
                    self.Notify("!!!-------> Accuracy || Loss  <-------")
                    CalcAverage("Training_Accuracy", k, "!!!", "Training_Loss")
                    if memory_rem < 500 and self.BatchSize > 1:
                        self.BatchSize -= 1
                    del train_loader 

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

            self.Stats["EpochTime"].append(time.time() - TimeStartEpoch)
            self.Notify("! >========= DURATION: " + str(datetime.timedelta(seconds = self.Stats["EpochTime"][-1])))
            self.DumpStatistics()

            self.ExportModel(valid_loader)

        self.Stats["TrainingTime"] = time.time() - TimeStart
        self.Stats.update(self.FileTraces)
        
        self.Stats["n_Node_Files"] = [[] for i in range(len(self.Stats["Start"]))]
        self.Stats["n_Node_Count"] = [[] for i in range(len(self.Stats["Start"]))]
        self.TrainingSample = [smpl for node in self.TrainingSample for smpl in self.RecallFromCache(self.TrainingSample[node], self.CacheDir)]       
        for s_i in range(len(self.TrainingSample)):
            smpl = self.TrainingSample[s_i]
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
        self.epoch = "Done"
        self.DumpStatistics()

