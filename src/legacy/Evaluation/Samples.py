from AnalysisTopGNN.Generators import GenerateDataLoader, EventGenerator
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.IO import UnpickleObject
from Tooling import Tools
import random
import os

class Sample(GenerateDataLoader):

    def __init__(self):
        self.src = None
        self.dst = None
        self.prc = None
        self.hash = None
        self.index = None
        self.train = None
        self.Data = None
    
    def Compile(self):
        try:
            os.symlink(self.src + "/" + self.hash + ".hdf5", self.dst + "/" + self.hash + ".hdf5") 
        except FileExistsError:
            pass
        self.Data = self.RecallFromCache(self.hash, self.dst)
        del self.src
        del self.dst

class SampleContainer(Tools, EventGenerator):

    def __init__(self):
        self.DataCache = None
        self.FileTrace = None
        self.TrainingSample = None
        self.HDF5 = None
        self.Threading = 12
        self.chnk = 1000
        self.random = False
        self.Size = 10
        self.Device = "cuda"

    def Collect(self, Rebuild = True):
        self.FileTr = UnpickleObject(self.FileTrace)
        self.TrainingSmpl = UnpickleObject(self.TrainingSample)
        self.SampleMap = self.FileTr["SampleMap"]

        self.FileEventIndex = ["/".join(smpl.split("/")[-2:]).replace(".root", "") for smpl in self.FileTr["Samples"]]
        self.FileEventIndex = {self.FileEventIndex[i] : [self.FileTr["Start"][i], self.FileTr["End"][i]] for i in range(len(self.FileEventIndex))}
        self.ReverseHash = {str(j) : i for i in self.ListFilesInDir(self.DataCache) for j in UnpickleObject(self.DataCache + "/" + i + "/" + i + ".pkl").values()}
        
        if Rebuild:
            smpl = list(self.SampleMap) 
        else:
            rev = {self.SampleMap[i] : i for i in self.SampleMap}
            smpl = [rev[i.replace(".hdf5", "")] for i in self.ListFilesInDir(self.HDF5)]
            if len(smpl) == 0:
                return self.Collect(True)
        
        if self.random: 
            random.shuffle(smpl)
        
        self.SampleMap = {smpl[s] : self.SampleMap[smpl[s]] for s in range(int(len(smpl)*float(self.Size/100)))}

    def MakeSamples(self):
        mode = {i : True for i in self.UnNestDict(self.TrainingSmpl["Training"])}
        mode |= {i : False for i in self.UnNestDict(self.TrainingSmpl["Validation"])}
        
        for indx in self.SampleMap:
            smpl = Sample()
            smpl.index = indx
            smpl.hash = self.SampleMap[indx]

            file = self.EventIndexFileLookup(indx)
            sub = file.split("/")
            smpl.prc = sub.pop(0)
            
            smpl.train = mode[indx]

            dch = self.ReverseHash[smpl.hash]
            smpl.src = self.DataCache + "/" + dch + "/" + "/".join(sub)
            smpl.dst = self.HDF5
            
            self.SampleMap[indx] = smpl

        del self.ReverseHash
        del self.DataCache
        del self.FileEventIndex
    
    def Compile(self):
        def Function(inpt):
            for i in inpt:
                i.Compile()
            return inpt

        TH = Threading(list(self.SampleMap.values()), Function, self.Threading, self.chnk)
        TH.Start()
        for i in TH._lists:
            self.SampleMap[i.index] = i
            self.SampleMap[i.index].Data.to(self.Device)


