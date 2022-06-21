from Closure.GenericFunctions import * 
from Functions.GNN.Models.BaseLine import *
from Functions.GNN.Models.PDFNet import *
import Functions.FeatureTemplates.ParticleGeneric.EdgeFeature as ef
import Functions.FeatureTemplates.ParticleGeneric.NodeFeature as nf
import Functions.FeatureTemplates.ParticleGeneric.GraphFeature as gf

def TestBaseLine(Files, Names, CreateCache):

    Features = {}
    Features |= {"NT_" + i : j for i, j in zip(["Index"], [nf.Index])}
    Features |= {"NF_" + i : j for i, j in zip(["Index"], [nf.Index])}
    
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
    #Truth Features
    Features |= {"ET_" + i : j for i, j in zip(["Topo"], [ef.Index])}
    Features |= {"NT_" + i : j for i, j in zip(["Index"], [nf.Index])}
    Features |= {"GT_" + i : j for i, j in zip(["mu_actual", "nTops"], [gf.mu_actual, gf.nTops])}

    #Measured Features
    Features |= {"NF_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
    Features |= {"GF_" + i : j for i, j in zip(["mu", "met", "met_phi", "pileup", "nTruthJet"], 
                                               [gf.mu, gf.met, gf.met_phi, gf.pileup, gf.nTruthJet])}
    CreateCache = True
    DL = CreateModelWorkspace(Files, Features, CreateCache, 100, Names, "TruthTopChildren")
    samples = DL.TrainingSample
    samples = samples[max(list(samples))][:10]
   
    Model = BaseLineModelEvent()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.0001
    Op.DefineOptimizer()
    
    kill = {}
    kill |= {"Topo" : "C", "Index" : "C", "mu_actual" : "R", "nTops" : "C"}
    KillCondition(kill, 100, Op, samples, 10000, 0.5, batched = 2)

    return True

def TestPDFNet(Files, Names, CreateCache):

    #Measured Features
    Features = {}
    Features |= {"EP_" + i : j for i, j in zip(["pT"], [ef.Expected_pT])}
    Features |= {"NF_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
    
    # Fake truth - Observables
    Features |= {"NT_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
    
    # Real Truth 
    Features |= {"NT_" + i : j for i, j in zip(["expPT"], [nf.ExpPT])}   





    # Create a model just for the TruthTopChildren 
    CreateCache = False
    DL = CreateModelWorkspace(Files, Features, CreateCache, 100, Names, "TruthTopChildren")
    samples = DL.TrainingSample
    samples = samples[max(list(samples))][:4]
   
    Model = PDFNetTruthChildren()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.0001
    Op.DefineOptimizer()

    kill = {}
    kill |= {"eta" : "R", 
             "energy" : "R", 
             "pT" : "R", 
             "phi" : "R"}
    KillCondition(kill, 100, Op, samples, 10000, sleep = 1)


    return True
