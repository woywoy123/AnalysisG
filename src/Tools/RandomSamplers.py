import random
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

from AnalysisTopGNN.Notification import RandomSamplers
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

    def MakekFolds(self, sample, folds, batch_size = 1, shuffle = True):
        smpl = len(sample)
        if len(sample) < folds:
            return 

        split = KFold(n_splits = folds, shuffle=shuffle)
        output = {}
        for fold, (train_idx, test_idx) in enumerate(split.split(np.arange(len(sample)))):
            output[fold] = []
            output[fold] += [DataLoader(sample, batch_size=batch_size, sampler = SubsetRandomSampler(train_idx))]
            output[fold] += [DataLoader(sample, batch_size=batch_size, sampler = SubsetRandomSampler(test_idx))]
        return output
