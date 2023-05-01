import random
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from torch.utils.data import SubsetRandomSampler
from AnalysisG.Notification import _RandomSamplers
from AnalysisG.Settings import Settings
from torch_geometric.loader import DataListLoader

class RandomSamplers(_RandomSamplers, Settings):

    def __init__(self):
        self.Caller = "RANDOMSAMPLER"
        Settings.__init__(self)
    
    def RandomizeEvents(self, Events, nEvents):
        if isinstance(Events, dict): Indx = list(Events.values())
        else: Indx = Events
        random.shuffle(Indx)
        if len(Events) >= nEvents: return Indx[0:nEvents]
        self.NotEnoughEvents(Events, nEvents)
        return {i.hash : i for i in Indx}   

    def MakeTrainingSample(self, Sample, TrainingSize = 50):
        if isinstance(Sample, dict) == False: return self.ExpectedDictionarySample(type(Sample))
        All = np.array(list(Sample))
        self.RandomizingSamples(len(All), TrainingSize)
        rs = ShuffleSplit(n_splits = 1, test_size = float((100 - TrainingSize)/100), random_state = 42)
        for train_idx, test_idx in rs.split(All): pass
        
        for i in All[train_idx]: Sample[i].TrainMode = "train" 
        for i in All[test_idx]: Sample[i].TrainMode = "test" 
        return {"train_hashes" : All[train_idx], "test_hashes" : All[test_idx]}

    def MakekFolds(self, sample, folds, batch_size = 1, shuffle = True, asHashes = False):
        if isinstance(sample, dict):
            smpl = [i for i in sample.values()]
            try: sample = {i : sample[i] for i in sample if sample[i].TrainMode != "test"}
            except AttributeError: pass
            sample = list(sample) if asHashes else list(sample.values()) 
        elif isinstance(sample, list): 
            try: sample = [i.hash if asHashes else i for i in sample if i.TrainMode != "test"]
            except AttributeError: pass
        else: return False
        if len(sample) < folds: return False
        sample = np.array(sample)
        split = KFold(n_splits = folds, shuffle=shuffle)
        output = {"k-" + str(f+1) : {"train" : None, "leave-out" : None} for f in range(folds)}
        for f, (train_idx, test_idx) in enumerate(split.split(np.arange(len(sample)))):
            output["k-" + str(f+1)]["train"] = sample[train_idx].tolist() 
            output["k-" + str(f+1)]["leave-out"] = sample[test_idx].tolist() 
        return output

    def MakeDataLoader(self, sample, SortByNodes = False, batch_size = 1):
        if isinstance(sample, dict): sample = list(sample.values())
        elif isinstance(sample, list): pass
        else: return False 
        
        output = {} 
        if SortByNodes: 
            for i in sample: output[i.num_nodes.item()] = [i] if i.num_nodes.item() not in output else [i]
        else: output["all"] = sample
        for i in output: output[i] = DataListLoader(output[i], batch_size=batch_size)
        return output
