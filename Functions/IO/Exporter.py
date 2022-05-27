import torch
import onnx
import numpy as np
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

    def __ProcessSample(self):
        if "dataloader" in str(type(self.__Sample)):
            self.__Sample = [i for i in self.__Sample][0].to_data_list()[0].detach().to_dict()
            self.__Sample = [self.__Sample[i] for i in self.__InputMap]
            return 
        

    def ExportEventGenerator(self, EventGenerator, Name = None, OutDirectory = None, DumpName = None):
        if Name == None:
            Name = "UNTITLED"
        if OutDirectory == None:
            OutDirectory = "_Pickle/"
       
        self.__Dump = HDF5(OutDirectory, Name)
        self.__Dump.StartFile()
        self.__Dump.DumpObject(EventGenerator, DumpName)
        
        for i in EventGenerator.Events:
            event = EventGenerator.Events[i]
            for k in event:
                ev = event[k]
                self.__Dump.DumpObject(ev)
        self.__Dump.EndFile()
    
    def ImportEventGenerator(self, Name = None, InputDirectory = None, DumpName = None):
        self.__Collect = HDF5()
        self.__Collect.OpenFile(InputDirectory, Name)
        return self.__Collect.RebuildObject(DumpName)


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

        self.__ProcessSample()
        self.Model.eval()
        self.Model.requires_grad_(False)
        if self.ONNX_Export:
            self.__ExportONNX( self.__OutputDir + "ONNX" + self.__EpochDir + ".onnx")
        if self.TorchScript_Export:
            self.__ExportTorchScript( self.__OutputDir + "TorchScript" + self.__EpochDir + ".pt")
        self.Model.requires_grad_(True)

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


