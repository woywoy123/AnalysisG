import torch
from Functions.IO.Files import WriteDirectory

class ExportToDataScience(WriteDirectory):
    def __init__(self):
        WriteDirectory.__init__(self)
        self.Model = None
        

    def ConvertSample(self, Sample):
        
        self.Sample = [i for i in DummySample][0]



        pass

    def ExportONNX(self, DummySample, Name):
        import onnx
        pass

    def ExportTorchScript(self, DummySample, Name):
        pass









    '''
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
    
    '''
