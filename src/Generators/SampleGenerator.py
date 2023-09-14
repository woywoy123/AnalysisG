from sklearn.model_selection import ShuffleSplit, KFold
from AnalysisG.Notification import _RandomSamplers
from torch_geometric.loader import DataListLoader
from torch.utils.data import SubsetRandomSampler
from AnalysisG.SampleTracer import SampleTracer
import numpy as np
import random

class RandomSamplers(_RandomSamplers, SampleTracer):
    def __init__(self):
        SampleTracer.__init__(self)
        self.Caller = "RANDOMSAMPLER"

    def RandomizeEvents(self, Events, nEvents = None):
        if isinstance(Events, list): Indx = Events
        else: Indx = list(Events.values())
        if nEvents is None: nEvents = len(Indx)

        random.shuffle(Indx)
        if len(Events) >= nEvents:
            return {i.hash: i for i in Indx[0:nEvents]}
        self.NotEnoughEvents(Events, nEvents)
        return {i.hash: i for i in Indx}

    def MakeTrainingSample(self, Sample, TrainingSize=50):
        if isinstance(Sample, list): Sample = {i.hash : i for i in Sample}
        elif self.is_self(Sample): Sample = {i.hash : i for i in Sample if i.Graph}
        elif isinstance(Sample, dict): pass
        else: return self.ExpectedDictionarySample(type(Sample))

        All = np.array(list(Sample))
        self.RandomizingSamples(len(All), TrainingSize)
        rs = ShuffleSplit(
            n_splits=1, test_size=float((100 - TrainingSize) / 100), random_state=42
        )
        for train_idx, test_idx in rs.split(All): pass
        for i in All[train_idx]: Sample[i].Train = True
        for i in All[test_idx]:  Sample[i].Eval  = True
        self.RandomizingSize(len(train_idx), len(test_idx))

        return {
            "train_hashes": All[train_idx].tolist(),
            "test_hashes": All[test_idx].tolist(),
        }

    def MakekFolds(self, sample, folds, batch_size=1, shuffle=True, asHashes=False):
        if isinstance(sample, dict):
            try: smpl = {i.hash : i for i in list(sample.values()) if not i.Eval}
            except AttributeError: return False
        elif isinstance(sample, list):
            try: smpl = {i.hash : i for i in sample if not i.Eval}
            except AttributeError: return False
        else: return False
        sample = list(smpl)
        if len(sample) < folds: return False

        split = KFold(n_splits=folds, shuffle=shuffle)
        output = {}
        sample = np.array(sample)
        for f, (train_idx, test_idx) in enumerate(split.split(np.arange(len(sample)))):
            train_idx, test_idx = sample[train_idx].tolist(), sample[test_idx].tolist()

            key = "k-" + str(f + 1)
            output[key] = {}
            if asHashes: output[key]["train"] = train_idx
            else: output[key]["train"] = [smpl[i] for i in train_idx]
            if asHashes: output[key]["leave-out"] = test_idx
            else: output[key]["leave-out"] = [smpl[i] for i in test_idx]
        return output

    def MakeDataLoader(self, sample, SortByNodes=False, batch_size=1):
        if isinstance(sample, dict): sample = list(sample.values())
        elif isinstance(sample, list): pass
        elif self.is_self(sample): pass
        else: return False

        output = {}
        for i in sample:
            if not i.Graph: continue
            key = "all" if not SortByNodes else i.num_nodes
            if key not in output: output[key] = []
            output[key] += [i.hash]

        _all = []
        for i in output:
            data = DataListLoader(output[i], batch_size = batch_size)
            _all += [j for j in data]
        return _all
