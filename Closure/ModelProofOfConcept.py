from Closure.GenericFunctions import * 
from Functions.GNN.Models.BaseLine import *
import Closure.FeatureTemplates.EdgeFeatures as ef
import Closure.FeatureTemplates.NodeFeatures as nf
import Closure.FeatureTemplates.GraphFeatures as gf

def BaseLine(Files, Names, CreateCache):

    Features = {}
    Features |= {"NT_" + i : j for i, j in zip(["Index"], [nf.Index])}
    Features |= {"NF_" + i : j for i, j in zip(["Index"], [nf.Index])}

    CreateCache = False
    if CreateCache:
        DL = CreateModelWorkspace(Files, Features, CreateCache, -1, Names, "TruthTopChildren")
        samples = DL.TrainingSample
        samples = samples[max(list(samples))][:4]
   
        Model = BaseLineModel(1, 4)
        Op = OptimizerTemplate(DL, Model)
        Op.LearningRate = 0.0001
        Op.WeightDecay = 0.0001
        Op.DefineOptimizer()

        kill = {}
        kill |= {"Index" : "C"}
        KillCondition(kill, 100, Op, samples, 10000)

    Features = {}
    Features |= {"ET_" + i : j for i, j in zip(["Signal"], [ef.Signal])}
    Features |= {"GT_" + i : j for i, j in zip(["Signal"], [gf.Signal])}
    Features |= {"NF_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pt, nf.phi])}

    CreateCache = False
    DL = CreateModelWorkspace(Files, Features, CreateCache, -1, Names, "TruthTopChildren")
    samples = DL.TrainingSample
    samples = samples[max(list(samples))][:10]
   
    Model = BaseLineModelAdvanced()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.0001
    Op.DefineOptimizer()

    kill = {}
    kill |= {"Signal" : "C"}
    KillCondition(kill, 100, Op, samples, 10000, 0.1)

    return True
