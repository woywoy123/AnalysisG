import torch
import onnx
import numpy as np
import string
from Functions.IO.Files import WriteDirectory
from Functions.IO.IO import HDF5
import Functions

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

class ExportToDataScience(WriteDirectory):
    def __init__(self):
        self.Model = None
        self.ONNX_Export = None 
        self.TorchScript_Export = None
        self.RunDir = None
        self.RunName = None

        self.epoch = 0
        self.Epochs = 0

        self.Caller = "ExportToDataScience"
        self.__OutputDir = None
        self.__EpochDir = None

    def __Routing(self, val, MemoryLink = None):
        x = None
        if isinstance(val, dict):
            x = self.__RecursiveDict(val, MemoryLink)
        elif isinstance(val, list):
            x = self.__RecursiveList(val, MemoryLink)
        elif "Functions." in str(type(val)) and MemoryLink == None:
            self.__Dump.DumpObject(val)
            self.__RecursiveDump(val)
        elif isinstance(val, str):
            if val.startswith("0x"):
                return MemoryLink[val]
            elif "Functions." not in val:
                return x
            x = MemoryLink[val.split(" ")[-1].replace(">", "")]
        return x

    def __RecursiveList(self, lst, MemoryLink = None):
        val = [self.__Routing(i, MemoryLink) for i in lst]
        val = [i for i in val if i != None]
        if len(val) != 0:
            return val

    def __RecursiveDict(self, dct, MemoryLink = None):
        val = {i : self.__Routing(j, MemoryLink) for i, j in dct.items()}
        val = {i : j for i, j in val.items() if j != None}
        if len(val) != 0:
            return val

    def __RecursiveDump(self, Obj):
        for k, j in Obj.__dict__.items():
            self.__Routing(j)

    def __RecursiveMemoryLink(self, Obj, MemoryLink):
        for k, j in Obj.__dict__.items():
            v = self.__Routing(j, MemoryLink)
            if v != None:
                setattr(Obj, k, v)

    def __ExportONNX(self, Dir):
        self.MakeDir("/".join(Dir.split("/")[:-1]))
        
        torch.onnx.export(
                self.Model, 
                tuple(self.__Sample), 
                Dir, 
                verbose = False,
                export_params = True, 
                opset_version = 14,
                input_names = list(self.__InputMap), 
                output_names = [i for i in self.__OutputMap if i.startswith("O_")])

    def __ExportTorchScript(self, Dir):
        self.MakeDir("/".join(Dir.split("/")[:-1]))
        model = torch.jit.trace(self.Model, self.__Sample)
        torch.jit.save(model, Dir, _extra_files = self.__OutputMap)


    def ExportDataGenerator(self, DataGenerator, Name = None, OutDirectory = None):
        for i in DataGenerator.EdgeAttribute:
            DataGenerator.EdgeAttribute[i] = str(DataGenerator.EdgeAttribute[i]).split(" ")[1]

        for i in DataGenerator.NodeAttribute:
            DataGenerator.NodeAttribute[i] = str(DataGenerator.NodeAttribute[i]).split(" ")[1]

        for i in DataGenerator.GraphAttribute:
            DataGenerator.GraphAttribute[i] = str(DataGenerator.GraphAttribute[i]).split(" ")[1]

        if Name == None:
            Name = "UNTITLED"
        if OutDirectory == None:
            OutDirectory = "_Pickle/"

        DataGenerator.SetDevice("cpu")
        self.__Dump = HDF5(OutDirectory, Name)
        self.__Dump.StartFile()
        for i, j in DataGenerator.DataContainer.items():
            self.__Dump.DumpObject(j)
            DataGenerator.DataContainer[i] = str(hex(id(j)))
    
        for i, j in DataGenerator.TrainingSample.items():
            for k in range(len(j)):
                DataGenerator.TrainingSample[i][k] = str(hex(id(j[k])))
 
        for i, j in DataGenerator.ValidationSample.items():
            for k in range(len(j)):
                DataGenerator.ValidationSample[i][k] = str(hex(id(j[k])))
        self.__Dump.DumpObject(DataGenerator)
        self.__Dump.EndFile()

    def ImportDataGenerator(self, Name = None, InputDirectory = None):
        self.__Collect = HDF5()
        self.__Collect.OpenFile(InputDirectory, Name)
        Objects = self.__Collect.RebuildObject()
        Obj = {}
        for i, j in Objects.items():
            name = type(j).__name__
            if name not in Obj:
                Obj[name] = []
            Obj[name].append(j)
            self.__RecursiveMemoryLink(j, Objects)
       
        Obj["GenerateDataLoader"][0].SetDevice("cuda")
        self.__Collect.EndFile()
        return Obj["GenerateDataLoader"][0]

    def ExportEventGenerator(self, EventGenerator, Name = None, OutDirectory = None):

        if Name == None:
            Name = "UNTITLED"
        if OutDirectory == None:
            OutDirectory = "_Pickle/"
       
        self.__Dump = HDF5(OutDirectory, Name)
        self.__Dump.StartFile()
        self.__Dump.DumpObject(EventGenerator)
        self.__RecursiveDump(EventGenerator) 
        self.__Dump.EndFile()
    
    def ImportEventGenerator(self, Name = None, InputDirectory = None):

        self.__Collect = HDF5()
        self.__Collect.OpenFile(InputDirectory, Name)
        Objects = self.__Collect.RebuildObject()
        ObjectDict = {}
        for addr in Objects:
            i = Objects[addr]
            self.__RecursiveMemoryLink(i, Objects)
            x = str(type(i).__name__)
            if x not in ObjectDict:
                ObjectDict[x] = []
            ObjectDict[x].append(i)
        self.__Collect.EndFile()
        return ObjectDict["EventGenerator"][0]

    def ExportModel(self, Sample):
        WriteDirectory.__init__(self)
        self.__EpochDir = "/Epoch_" + str(self.epoch+1) + "_" + str(self.Epochs)
        self.__OutputDir = self.RunDir + "/" + self.RunName + "/"       
        self.__Sample = Sample 
       
        self.__InputMap = {}
        for i in self.ModelInputs:
            self.__InputMap[i] = str(self.ModelInputs.index(i))

        self.__OutputMap = {}
        for i in self.ModelOutputs:
            self.__OutputMap[i] = str(self.ModelOutputs[i]) + "->" + str(type(self.ModelOutputs[i])).split("'")[1]

        self.__Sample = [i for i in self.__Sample][0].to_data_list()[0].detach().to_dict()
        self.__Sample = [self.__Sample[i] for i in self.__InputMap]
        self.Model.eval()
        self.Model.requires_grad_(False)
        if self.ONNX_Export:
            self.__ExportONNX( self.__OutputDir + "ONNX" + self.__EpochDir + ".onnx")
        if self.TorchScript_Export:
            self.__ExportTorchScript( self.__OutputDir + "TorchScript" + self.__EpochDir + ".pt")
        self.Model.requires_grad_(True)




