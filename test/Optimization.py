from AnalysisTopGNN.Tools import Tools
import torch

class Epoch:

    def __init__(self):
        pass


class LossFunctions:

    def __init__(self):
        pass

class Scheduler:

    def __init__(self):
        pass


class Optimizer:
    
    def __init__(self):
        pass

class ModelTrainer(Tools):

    def __init__(self):
        self.Model = None
        self.Device = None
        self.Scheduler = None
        self.Optimizer = None
        self.BatchSize = None
        self.SplitSampleByNode = False
        self.Samples = {}
        self.Tree = None
 
    def AddAnalysis(self, analysis):
        for smpl in analysis:
            num_nodes = smpl.Trees[self.Tree].num_nodes
            if num_nodes not in self.Samples:
                self.Samples[num_nodes] = []
            self.Samples[num_nodes].append(smpl)

