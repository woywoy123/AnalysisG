import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import accuracy

from sklearn.model_selection import KFold
import numpy as np

import time
from Functions.Tools.Alerting import Notification
from Functions.IO.Files import WriteDirectory, Directories
from Functions.IO.IO import PickleObject

class Optimizer(Notification):

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

        ### Internal Stuff 
        self.Training = True
        self.Sample = None
        self.T_Features = {}
      
    def ReadInDataLoader(self, DataLoaderInstance):
        self.TrainingSample = DataLoaderInstance.TrainingSample

        self.EdgeFeatures = DataLoaderInstance.EdgeAttribute
        self.NodeFeatures = DataLoaderInstance.NodeAttribute
        self.GraphFeatures = DataLoaderInstance.GraphAttribute

        self.Device_S = DataLoaderInstance.Device_S
        self.Device = DataLoaderInstance.Device
        self.DataLoader = DataLoaderInstance

    def DumpStatistics(self):
        WriteDirectory().MakeDir(self.RunDir + "/" + self.RunName + "/Statistics")
        if self.epoch == "Done":
            PickleObject(self.Stats, "Stats_" + self.epoch, self.RunDir + "/" + self.RunName + "/Statistics")
        else:
            PickleObject(self.Stats, "Stats_" + str(self.epoch+1), self.RunDir + "/" + self.RunName + "/Statistics")
        self.MakeStats()

    def MakeStats(self):

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

    def __GetFlags(self, inp, FEAT):
        if FEAT == "M":
            self.ModelInputs = list(self.Model.forward.__code__.co_varnames[:self.Model.forward.__code__.co_argcount])
            self.ModelInputs.remove("self")
            
            input_len = len(list(set(list(inp.__dict__["_store"])).intersection(set(self.ModelInputs))))
            if "batch" in self.ModelInputs:
                input_len += 1

            if input_len < len(self.ModelInputs):
                
                self.Warning("---> ! Features Expected in Model Input ! <---")
                for i in self.ModelInputs:
                    self.Warning("+-> " + i)
                
                self.Warning("---> ! Features Found in Sample Input ! <---")
                for i in list(inp.__dict__["_store"]):
                    self.Warning("+-> " + i)
                
                self.Fail("MISSING VARIABLES IN GIVEN DATA SAMPLE")

            self.Notify("FOUND ALL MODEL INPUT PARAMETERS IN SAMPLE")
            for i in self.ModelInputs:
                self.Notify("---> " + i)

            Setting = [i for i in self.Model.__dict__ if i.startswith("C_") or i.startswith("L_") or i.startswith("O_") or i.startswith("N_")]
            self.ModelOutputs = {}
            for i in Setting:
                self.ModelOutputs[i] = self.Model.__dict__[i]
            return
            
        for i in inp:
            if i.startswith("T_"):
                self.T_Features[i[2:]] = [FEAT + "_" +i, FEAT + "_" +i[2:]]

    def __ExportONNX(self, DummySample, Name):
        import onnx
        
        DummySample = tuple([DummySample[i] for i in self.ModelInputs])
        torch.onnx.export(
                self.Model, DummySample, Name,
                export_params = True, 
                input_names = self.ModelInputs, 
                output_names = [i for i in self.ModelOutputs if i.startswith("O_")])

   
    def __ExportTorchScript(self, DummySample, Name):
        DummySample = tuple([DummySample[i] for i in self.ModelInputs])
      
        Compact = {}
        for i in self.ModelInputs:
            Compact[i] = str(self.ModelInputs.index(i))

        p = 0
        for i in self.ModelOutputs:
            if i.startswith("O_"):
                Compact[i] = str(p)
                p+=1
            else:
                Compact[i] = str(self.ModelOutputs[i])

        model = torch.jit.trace(self.Model, DummySample)
        torch.jit.save(model, Name, _extra_files = Compact)

    def __ImportTorchScript(self, Name):
        class Model:
            def __init__(self, dict_in, model):
                self.__Model = model
                self.__router = {}
                for i in dict_in:
                    setattr(self, i, dict_in[i])
                    if i.startswith("O_"):
                        self.__router[dict_in[i]] = i         
            
            def __call__(self, **kargs):
                pred = list(self.__Model(**kargs))
                for i in range(len(pred)):
                    setattr(self, self.__router[i], pred[i])

            def train(self):
                self.__Model.train(True)

            def eval(self):
                self.__Model.train(False)
        
        extra_files = {}
        for i in list(self.ModelOutputs):
            extra_files[i] = ""
        for i in list(self.ModelInputs):
            extra_files[i] = ""
        
        M = torch.jit.load(Name, _extra_files = extra_files)
        for i in extra_files:
            conv = str(extra_files[i].decode())
            if conv.isnumeric():
                conv = int(conv)
            if conv == "True":
                conv = True
            if conv == "False":
                conv = False
            extra_files[i] = conv
         
        self.Model = Model(extra_files, M)
    
    def __SaveModel(self, DummySample):
        self.Model.eval()
        DummySample = [i for i in DummySample][0]
        DummySample = DummySample.to_data_list()[0].detach().to_dict()
        DirOut = self.RunDir + "/" + self.RunName + "/"
        if self.ONNX_Export:
            WriteDirectory().MakeDir(DirOut + "ModelONNX")
            Name = DirOut + "ModelONNX/Epoch_" + str(self.epoch+1) + "_" + str(self.Epochs) + ".onnx"
            self.__ExportONNX(DummySample, Name)
        
        if self.TorchScript_Export:
            WriteDirectory().MakeDir(DirOut + "ModelTorchScript")
            Name = DirOut + "ModelTorchScript/Epoch_" + str(self.epoch+1) + "_" + str(self.Epochs) + ".pt"
            self.__ExportTorchScript(DummySample, Name)

    def DefineOptimizer(self):
        self.Model.to(self.Device)
        if self.DefaultOptimizer == "ADAM":
            self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        elif self.DefaultOptimizer == "SGD":
            self.Optimizer = torch.optim.SGD(self.Model.parameters(), lr = self.LearningRate)

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
                key_T = self.T_Features[key][0]
                out_v = sample[key_T]
            else:
                out_v = self.Model.__dict__["O_" + key]
            out_p = out_v

            # Adjust the outputs
            if GetKeyPair(output_dict, "N_" + key):
                out_v = out_v[sample.edge_index[0]]
                out_p = out_v
            
            if GetKeyPair(output_dict, "C_" + key) and not Truth:
                out_p = out_p.max(1)[1]
            
            out_p = out_p.view(1, -1)[0]
            OutDict[key] = [out_p, out_v]
        return OutDict


    def Train(self, sample):
        if self.Training:
            self.Model.train()
            self.Optimizer.zero_grad()
        else:
            self.Model.eval()

        self.MakePrediction(sample)
        truth_out = self.Output(self.ModelOutputs, sample, Truth = True)
        model_out = self.Output(self.ModelOutputs, sample, Truth = False)
        
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
                self.Fail("Your Classification Model only has " + str(int(m_v.shape[1])) + " classes but requires " + str(int(t_v.max()+1)))
            elif Classification == False and m_v.shape[1] != t_v.shape[1]:
                self.Fail("Your Model has more outputs than Truth! :: " + key)
            
            acc = acc(t_p, m_p)
            L = self.LF(m_v, t_v)
            if self.Training:
                self.Stats["Training_Accuracy"][key][-1].append(acc)
                self.Stats["Training_Loss"][key][-1].append(L)
            else:
                self.Stats["Validation_Accuracy"][key][-1].append(acc)
                self.Stats["Validation_Loss"][key][-1].append(L)
            LT += L

        if self.Training:
            LT.backward()
            self.Optimizer.step()

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

    def KFoldTraining(self):
        def CalcAverage(Mode, k, Notify = "", Mode2 = None):
            for f_k in self.Stats[Mode]:
                v_1 = str(format(float(sum(self.Stats[Mode][f_k][k])/len(self.Stats[Mode][f_k][k])), ' >10.4g'))
                v_2 = str(format(float(sum(self.Stats[Mode2][f_k][k])/len(self.Stats[Mode2][f_k][k])), ' >10.4g'))
                    
                self.Notify(Notify + "       " + str(v_1) + " || " + str(v_2) + " (" + f_k + ")")


        self.DefineOptimizer()
        Splits = KFold(n_splits = self.kFold, shuffle = True, random_state= 42)
        N_Nodes = list(self.TrainingSample)
        N_Nodes.sort(reverse = True)
        self.__GetFlags(self.EdgeFeatures, "E")
        self.__GetFlags(self.NodeFeatures, "N")
        self.__GetFlags(self.GraphFeatures, "G")
        self.__GetFlags(self.TrainingSample[N_Nodes[0]][0], "M")

        self.MakeStats()
        self.Model.Device = self.Device_S
        
        TimeStart = time.time()
        for self.epoch in range(self.Epochs):
            self.Notify("! >============== [ EPOCH (" + str(self.epoch+1) + "/" + str(self.Epochs) + ") ] ==============< ")
            
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

                    train_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(train_idx))
                    valid_loader = DataLoader(Curr, batch_size = self.BatchSize, sampler = SubsetRandomSampler(val_idx)) 

                    self.Training = True
                    self.SampleLoop(train_loader)
                    self.Notify("!!!-----++> Training <++-----")
                    self.Notify("!!!-------> Accuracy || Loss  <-------")
                    CalcAverage("Training_Accuracy", k, "!!!", "Training_Loss")


                    self.Training = False
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
            self.DumpStatistics()
            self.__SaveModel(train_loader)

        self.Stats["TrainingTime"] = time.time() - TimeStart
        self.Stats.update(self.DataLoader.FileTraces)
        
        self.Stats["n_Node_Files"] = [[] for i in range(len(self.Stats["Start"]))]
        self.Stats["n_Node_Count"] = [[] for i in range(len(self.Stats["Start"]))]
        self.TrainingSample = [smpl for node in self.TrainingSample for smpl in self.TrainingSample[node]]       
        for s_i in range(len(self.TrainingSample)):
            smpl = self.TrainingSample[s_i]
            indx, n_nodes = smpl.i, smpl.num_nodes
            find = [indx >= start and indx <= end for start, end in zip(self.Stats["Start"], self.Stats["End"])].index(True)
            
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

