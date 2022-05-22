import torch
import onnx
import h5py
import numpy as np
from Functions.IO.Files import WriteDirectory
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
        
        if isinstance(self.__Sample, Functions.Event.Event.Event):
             
            for i in self.__Sample.__dict__:
                if isinstance(self.__Sample.__dict__[i], list):
                    continue
                self.__GeneratorDump[self.__dump_e + "/E-" + i] = self.__Sample.__dict__[i]
            return 

        if issubclass( type(self.__Sample) , Functions.Particles.Particles.Particle):
            for i in self.__Sample.__dict__:
                if isinstance(self.__Sample.__dict__[i], list):
                    continue

                if str(self.__dump_e + "/" + i) not in self.__GeneratorDump:
                    self.__GeneratorDump[self.__dump_e + "/" + i] = []
                self.__GeneratorDump[self.__dump_e + "/" + i].append(self.__Sample.__dict__[i])



    def ExportEventGenerator(self, EventGenerator):
        self.__FileDictionary = EventGenerator.FileEventIndex
        self.__GeneratorDump = {}
        self.__GeneratorDump["Files"] = []

        for ev_i in EventGenerator.Events:
            F = EventGenerator.EventIndexFileLookup(ev_i) 
            if F not in self.__GeneratorDump["Files"]:
                self.__GeneratorDump["Files"].append(F)

            idx = "F-" + str(self.__GeneratorDump["Files"].index(F))
            dump_s = idx + "/i-" + str(ev_i)

            for key in EventGenerator.Events[ev_i]:
                EventObject =  EventGenerator.Events[ev_i][key]

                self.__dump_e = dump_s + "/T-" + key
                self.__Sample = EventObject
                self.__ProcessSample()
                 
                for p in EventObject.__dict__:
                    if isinstance(EventObject.__dict__[p], list) == False:
                        continue
                    for k in EventObject.__dict__[p]:
                        self.__Sample = k
                        self.__ProcessSample()
                
            
        for i in self.__GeneratorDump:
            print(i) # <---- continue here and write the hdf5 dump.


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


