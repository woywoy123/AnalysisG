from AnalysisTopGNN.Tools.ModelTesting import CreateWorkspace, KillCondition, OptimizerTemplate

from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton
from EventFeatureTemplate import ApplyFeatures, TruthJets

from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from PathNets import PathNetsTruthJet
import torch

from PathNetOptimizer import PathCombinatorial

def PathNetsDEBUG():
    GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
    Files = [GeneralDir + "t", GeneralDir + "ttbar", GeneralDir + "tttt"]
    Names = ["tttt", "ttbar", "t"]
    
    CreateCache = False
    Features = TruthJets()
    DL = CreateWorkspace(Files, Features, CreateCache, 100, Names, "TruthJetLepton", True)
    samples = DL.TrainingSample

    Model = PathNetsTruthJet()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.001
    Op.DefineOptimizer()
    Op.Debug = True

    #PickleObject(samples[19][0], "19")
    k = 19
    kill = {}
    kill |= {"edge" : "C"}
    #kill |= {"from_res" : "C"}
    #kill |= {"signal_sample": "C"}
    #kill |= {"from_top": "C"}
    KillCondition(kill, 50, Op, samples[k], 100000, sleep = 2, batched = 1)

def CombinationTest():
    #smpl = UnpickleObject("19")
    #print(smpl)
    x = PathCombinatorial(32, 3, "cuda")
    
    import time 
    time.sleep(100)



if __name__ == "__main__":
    #CombinationTest()
    PathNetsDEBUG()
