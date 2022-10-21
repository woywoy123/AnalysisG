
#from AnalysisTopGNN.Tools import Notification 
#from AnalysisTopGNN.IO import ExportToDataScience
#from AnalysisTopGNN.Parameters import Parameters

#import numpy as np
#import importlib

from AnalysisTopGNN.Notification import RandomSamplers
import random
import numpy as np
from sklearn.model_selection import ShuffleSplit

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

    def MakeTrainingSample(self, Sample, TrainingSize = 50):
        if isinstance(Sample, dict) == False:
            self.ExpectedDictionarySample(type(Sample))
            return 
        
        All = np.array(list(Sample))
        self.RandomizingSamples(len(All), TrainingSize)
        
        rs = ShuffleSplit(n_splits = 1, test_size = float((100 - TrainingSize)/100), random_state = 42)
        for train_idx, test_idx in rs.split(All):
            pass 
        return {"train_hashes" : All[train_idx], "test_hashes" : All[test_idx]}



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
            


