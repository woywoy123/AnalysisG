
#from AnalysisTopGNN.Tools import Notification 
#from AnalysisTopGNN.IO import ExportToDataScience
#from AnalysisTopGNN.Parameters import Parameters

#from sklearn.model_selection import ShuffleSplit
#import numpy as np
#import importlib


from AnalysisTopGNN.Notification import RandomSamplers
import random

class RandomSamplers(RandomSamplers):

    def __init__(self):
        pass
    
    def RandomEvents(self, Events, nEvents):
        Indx = []
        if isinstance(Events, dict):
            Indx += list(Events.values())
        else:
            Indx += Events
        
        random.shuffle(Indx)
        if len(Events) < nEvents:
            self.NotEnoughEvents(Events, nEvents)
            return Indx    
        return Indx[0:nEvents]

    #def MakeTrainingSample(self, TrainingSize = None):
         



    #def MakeTrainingSample(self, ValidationSize = None):
    #    def MakeSample(Shuff, InputList):
    #        if isinstance(Shuff, int):
    #            Shuff = self.DataContainer
    #        for i in Shuff:
    #            n_p = int(self.DataContainer[i].num_nodes)
    #            if n_p == 0:
    #                continue
    #            if n_p not in InputList:
    #                InputList[n_p] = []
    #            InputList[n_p].append(self.DataContainer[i])
    #    
    #    if ValidationSize != None:
    #        self.ValidationSize = ValidationSize

    #    self.ProcessSamples() 
    #    self.Notify("!WILL SPLIT DATA INTO TRAINING/VALIDATION (" + 
    #            str(self.ValidationSize) + "%) - TEST (" + str(100 - self.ValidationSize) + "%) SAMPLES")

    #    All = np.array(list(self.DataContainer))
    #    
    #    if self.ValidationSize > 0 and self.ValidationSize < 100:
    #        rs = ShuffleSplit(n_splits = 1, test_size = float((100-self.ValidationSize)/100), random_state = 42)
    #        for train_idx, test_idx in rs.split(All):
    #            pass
    #        MakeSample(train_idx, self.TrainingSample)
    #        MakeSample(test_idx, self.ValidationSample)
    #    else:
    #        MakeSample(-1, self.TrainingSample)

    #def RecallFromCache(self, SampleList, Directory):

    #    def function(inpt):
    #        exp = ExportToDataScience()
    #        exp.VerboseLevel = 0
    #        out = []
    #        for i in inpt:
    #            out += list(exp.ImportEventGraph(i, self.DataCacheDir).values())
    #        return out

    #    if Directory == None:
    #        return SampleList
    #    
    #    if isinstance(SampleList, str):
    #        Exp = ExportToDataScience()
    #        Exp.VerboseLevel = 0
    #        dic = Exp.ImportEventGraph(SampleList, Directory)
    #        return dic[list(dic)[0]]
    #    elif isinstance(SampleList, list) == False: 
    #        self.Fail("WRONG SAMPLE INPUT! Expected list, got: " + type(SampleList))
    #    
    #    TH = Threading(SampleList, function, self.Threads, self.chnk)
    #    TH.VerboseLevel = self.VerboseLevel
    #    TH.Caller = self.Caller
    #    TH.Start()
    #    self.SetDevice(self.Device, TH._lists)
    #    return TH._lists
            


