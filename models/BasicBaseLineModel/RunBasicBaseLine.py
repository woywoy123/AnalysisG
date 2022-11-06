from AnalysisTopGNN.Tools.ModelTesting import CreateWorkspace, KillCondition, OptimizerTemplate
from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from BasicBaseLine import *
from EventFeatureTemplate import TruthJets, TruthTopChildren

def BaseLineModelTruthJet(Files, Names, CreateCache):
    CreateCache = False
    Features = TruthJets()
    DL = CreateWorkspace(Files, Features, CreateCache, 100, Names, "TruthJetLepton", True)
    samples = DL.TrainingSample
    k = 14 
    #su = 0
    #for i in samples:
    #    su += len(samples[i])
    #    print(i, len(samples[i]))
    #print(su)
    #exit()

    Model = BasicBaseLineTruthJet()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.01
    Op.WeightDecay = 0.001
    Op.DefineOptimizer()

    kill = {}
    kill |= {"edge" : "R"}
    #kill |= {"from_res" : "C"}
    #kill |= {"signal_sample": "C"}
    #kill |= {"from_top": "C"}
    KillCondition(kill, 50, Op, samples[k], 100000, sleep = 2, batched = 3)

def BasicBaseLineTruthChildren(Files, Names, CreateCache):
    Features = TruthTopChildren()
    DL = CreateWorkspace(Files, Features, CreateCache, 1000, Names, "TruthTopChildren", True)
    samples = DL.ValidationSample
 
    Model = BasicBaseLineRecursion()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.001
    Op.WeightDecay = 0.0001
    #Op.DefaultOptimizer = "SGD"
    Op.DefineOptimizer()


    kill = {}
    kill |= {"edge" : "C"}
    #kill |= {"from_res" : "C"}
    #kill |= {"signal_sample": "C"}
    #kill |= {"from_top": "C"}
    KillCondition(kill, 10, Op, samples, 100000, sleep = 2, batched = 4)


if __name__ == "__main__":
    GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
    Files = [GeneralDir + "ttbar/QU_0.root"]
    Names = ["tttt"]
    CreateCache = False
    #BaseLineModelTruthJet(Files, Names, CreateCache)
    BasicBaseLineTruthChildren(Files, Names, CreateCache)

