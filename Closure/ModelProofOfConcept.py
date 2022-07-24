from Closure.GenericFunctions import * 
from Functions.GNN.Models.BaseLine import *
from Functions.GNN.Models.PDFNet import *
from Functions.GNN.Models.BasicBaseLine import *
from Functions.GNN.TrivialModels import MassGraphNeuralNetwork
import Functions.FeatureTemplates.ParticleGeneric.EdgeFeature as ef
import Functions.FeatureTemplates.ParticleGeneric.NodeFeature as nf
import Functions.FeatureTemplates.ParticleGeneric.GraphFeature as gf

import Functions.FeatureTemplates.TruthTopChildren.NodeFeature as tc_nf

import Functions.FeatureTemplates.TruthJet.EdgeFeature as tj_ef
import Functions.FeatureTemplates.TruthJet.NodeFeature as tj_nf
import Functions.FeatureTemplates.TruthJet.GraphFeature as tj_gf


def AddFeature(Prefix, dic):
    return {Prefix + "_" + i : dic[i] for i in dic} 



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
    CreateCache = False
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
    Features |= {"NF_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
    
    # Fake truth - Observables
    Features |= {"NT_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
   
    # Truth Features
    Features |= {"ET_" + i : j for i, j in zip(["Index"], [ef.Index])}

    ## Real Truth 
    #Features |= {"NT_" + i : j for i, j in zip(["expPx"], [nf.ExpPx])}   
    ## Preprocessing 
    #Features |= {"EP_" + i : j for i, j in zip(["pT"], [ef.Expected_Px])}

    # Create a model just for the TruthTopChildren 
    CreateCache = False
    DL = CreateModelWorkspace(Files, Features, CreateCache, 100, Names, "TruthTopChildren")
    samples = DL.TrainingSample
    
    samples = [ i for k in samples for i in samples[k]]




    #samples = samples[max(list(samples))][:-1]
    
    ## Debug: Create a simple GNN that only looks at the mass 
    #Model = MassGraphNeuralNetwork() 
    #Op = OptimizerTemplate(DL, Model)
    #Op.LearningRate = 0.0001
    #Op.WeightDecay = 0.001
    #Op.DefineOptimizer()

    #kill = {}
    #kill |= {"Index" : "R"}
    #KillCondition(kill, 1000, Op, samples, 10000, sleep = 1)
    
    # ====== Experimental GNN stuff ======= #
    Model = GraphNeuralNetwork_MassTagger()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.001
    Op.WeightDecay = 0.001
    Op.DefineOptimizer()

    kill = {}
    kill |= {"Index" : "C"}
    KillCondition(kill, 1000, Op, samples, 100000, sleep = 2, batched = 2)
 


    #kill = {}
    #kill |= {"eta" : "R", 
    #         "energy" : "R", 
    #         "pT" : "R", 
    #         "phi" : "R",
    #         "Index" : "C"}
    #KillCondition(kill, 1000, Op, samples, 10000, sleep = 1, batched = 10)


    return True

def TestBasicBaseLine(Files, Names, CreateCache):
    
    ## =========================================== TRUTH CHILDREN GNN STUFF =========================================== #
    #Features = {}
    #Features |= {"NF_" + i : j for i, j in zip(["eta", "energy", "pT", "phi"], [nf.eta, nf.energy, nf.pT, nf.phi])}
    #
    ## Truth Features
    #Features |= {"NT_" + i : j for i, j in zip(["FromRes"], [tc_nf.FromRes])}
    #Features |= {"ET_" + i : j for i, j in zip(["Edge"], [ef.Index])}
    #Features |= {"GT_" + i : j for i, j in zip(["SignalSample"], [gf.SignalSample])}

    ## Create a model just for the TruthTopChildren 
    #CreateCache = True
    #DL = CreateModelWorkspace(Files, Features, CreateCache, 100, Names, "TruthTopChildren")
    #samples = DL.TrainingSample
    #
    #samples = [ i for k in samples for i in samples[k]]

    #Model = BasicBaseLineTruthChildren()
    #Op = OptimizerTemplate(DL, Model)
    #Op.LearningRate = 0.0001
    #Op.WeightDecay = 0.001
    #Op.DefineOptimizer()

    #kill = {}
    #kill |= {"Edge" : "C"}
    #kill |= {"FromRes" : "C"}
    #KillCondition(kill, 50, Op, samples, 100000, sleep = 2, batched = 2)
    
    # =========================================== TRUTH JET/ JET GNN STUFF =========================================== #
    # Create a model just for the TruthJets/Jets

    # Node: Generic Particle Properties
    GenPartNF = {
            "eta" : nf.eta, 
            "energy" : nf.energy, 
            "pT" : nf.pT, 
            "phi" : nf.phi, 
            "mass" : nf.mass, 
            "islep" : nf.islepton, 
            "charge" : nf.charge, 
    }

    # Graph: Generic Particle Properties
    GenPartGF = {
            "mu" : gf.mu, 
            "met" : gf.met, 
            "met_phi" : gf.met_phi, 
            "pileup" : gf.pileup, 
            "njets" : gf.nTruthJets, 
            "nlep" : gf.nLeptons
    }

    # Truth Edge: Truth Jet Properties
    TruthJetTEF = {
            "edge" : tj_ef.Index, 
    } 

    # Truth Node: Truth Jet Properties
    TruthJetTNF = {
            "tops_merged" : tj_nf.TopsMerged, 
            "from_top" : tj_nf.FromTop, 
    } 

    # Truth Node: Generic Paritcle Properties
    GenPartTNF = {
            "from_res" : nf.FromRes
    }

    # Truth Graph: Generic Paritcle Properties
    GenPartTGF = {
            "mu_actual" : gf.mu_actual,
            "nTops" : gf.nTops, 
            "signal_sample" : gf.SignalSample
    }

    Features = {}
    Features |= AddFeature("NF", GenPartNF)
    Features |= AddFeature("GF", GenPartGF)
    Features |= AddFeature("ET", TruthJetTEF)
    Features |= AddFeature("NT", TruthJetTNF)    
    Features |= AddFeature("NT", GenPartTNF)
    Features |= AddFeature("GT", GenPartTGF)

    CreateCache = True
    DL = CreateModelWorkspace(Files, Features, CreateCache, 100, Names, "TruthJetLepton", True)
    samples = DL.TrainingSample
    k = 19 
    #su = 0
    #for i in samples:
    #    su += len(samples[i])
    #    print(i, len(samples[i]))
    #print(su)
    #exit()

    Model = BasicBaseLineTruthJet()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.001
#    Op.DefaultOptimizer = "SGD"
    Op.DefineOptimizer()

    kill = {}
    kill |= {"edge" : "R"}
    #kill |= {"from_res" : "C"}
    #kill |= {"signal_sample": "C"}
    #kill |= {"from_top": "C"}
    KillCondition(kill, 50, Op, samples[k], 100000, sleep = 2, batched = 10)
 


    return True 
